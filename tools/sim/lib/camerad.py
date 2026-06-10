import numpy as np
import os
import pyopencl as cl
import pyopencl.array as cl_array

from cereal.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.common.basedir import BASEDIR
from openpilot.tools.sim.lib.common import W, H


def _rgb24_to_nv12_cpu(rgb: np.ndarray) -> bytes:
  # RGB -> YUV conversion (same coefficient family used by tools/lib/framereader.py)
  yuv_from_rgb = np.array([
    [0.25681631,  0.50415484,  0.09791402],
    [-0.14824553, -0.29102007,  0.4392656 ],
    [0.43927107, -0.36783273, -0.07143833],
  ])
  img = np.dot(rgb.reshape(-1, 3), yuv_from_rgb.T).reshape(rgb.shape)

  ys = img[:, :, 0]
  us = (img[::2, ::2, 1] + img[1::2, ::2, 1] + img[::2, 1::2, 1] + img[1::2, 1::2, 1]) / 4 + 128
  vs = (img[::2, ::2, 2] + img[1::2, ::2, 2] + img[::2, 1::2, 2] + img[1::2, 1::2, 2]) / 4 + 128

  y_len = rgb.shape[0] * rgb.shape[1]
  uv_len = y_len // 4

  nv12 = np.empty(y_len + 2 * uv_len, dtype=rgb.dtype)
  nv12[:y_len] = ys.reshape(-1)
  nv12[y_len::2] = us.reshape(-1)
  nv12[y_len + 1::2] = vs.reshape(-1)
  return nv12.clip(0, 255).astype(np.uint8).tobytes()

class Camerad:
  """Simulates the camerad daemon"""
  def __init__(self, dual_camera):
    self.pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState'])

    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 5, False, W, H)
    if dual_camera:
      self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 5, False, W, H)

    self.vipc_server.start_listener()

    # set up for pyopencl rgb to yuv conversion (fallback to CPU when unavailable)
    self.use_opencl = True
    try:
      self.ctx = cl.create_some_context()
      self.queue = cl.CommandQueue(self.ctx)
      cl_arg = f" -DHEIGHT={H} -DWIDTH={W} -DRGB_STRIDE={W * 3} -DUV_WIDTH={W // 2} -DUV_HEIGHT={H // 2} -DRGB_SIZE={W * H} -DCL_DEBUG "

      kernel_fn = os.path.join(BASEDIR, "tools/sim/rgb_to_nv12.cl")
      with open(kernel_fn) as f:
        prg = cl.Program(self.ctx, f.read()).build(cl_arg)
        self.krnl = prg.rgb_to_nv12
      self.Wdiv4 = W // 4 if (W % 4 == 0) else (W + (4 - W % 4)) // 4
      self.Hdiv4 = H // 4 if (H % 4 == 0) else (H + (4 - H % 4)) // 4
    except Exception as e:
      self.use_opencl = False
      print(f"[sim/camerad] OpenCL unavailable, using CPU RGB->NV12 conversion: {e}")

  def cam_send_yuv_road(self, yuv):
    self._send_yuv(yuv, self.frame_road_id, 'roadCameraState', VisionStreamType.VISION_STREAM_ROAD)
    self.frame_road_id += 1

  def cam_send_yuv_wide_road(self, yuv):
    self._send_yuv(yuv, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  # Returns: yuv bytes
  def rgb_to_yuv(self, rgb):
    assert rgb.shape == (H, W, 3), f"{rgb.shape}"
    assert rgb.dtype == np.uint8

    if not self.use_opencl:
      return _rgb24_to_nv12_cpu(rgb)

    rgb_cl = cl_array.to_device(self.queue, rgb)
    yuv_cl = cl_array.empty_like(rgb_cl)
    self.krnl(self.queue, (self.Wdiv4, self.Hdiv4), None, rgb_cl.data, yuv_cl.data).wait()
    yuv = np.resize(yuv_cl.get(), rgb.size // 2)
    return yuv.data.tobytes()

  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
    eof = int(frame_id * 0.05 * 1e9)
    self.vipc_server.send(yuv_type, yuv, frame_id, eof, eof)

    dat = messaging.new_message(pub_type, valid=True)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    self.pm.send(pub_type, dat)

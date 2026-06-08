#!/usr/bin/env python3
import argparse
import io
import time
from cereal import messaging
import numpy as np
import cv2
import av
from cereal import log
from pathlib import Path
from typing import Dict, List, Optional
from setproctitle import setproctitle
from cereal.messaging import PubMaster, SubMaster
import onnxruntime as ort
from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import config_realtime_process
from openpilot.selfdrive import sentry
from openpilot.selfdrive.modeld.runners import ModelRunner
from collections import deque


PROCESS_NAME = "selfdrive.modeld.scratchmodeld"

E2E_MODEL_PATHS = {
    ModelRunner.ONNX: Path(__file__).parent
    / "models/v2.2_lstm_optimized_for_tinygrad.onnx"
}

IMAGE_SIZE = 224
CAR_STATE_DIM = 5
PREDICTION_HORIZON = 10

# 車両状態キュー（過去12秒分、10Hz）
car_state_queue: deque = deque(maxlen=120)

def update_car_state_queue(car_state_data):
    timestamp = time.time()
    car_state_entry = {
        "timestamp": timestamp,
        "vEgo": car_state_data.get("vEgo", 0.0),
        "aEgo": car_state_data.get("aEgo", 0.0),
        "steeringAngleDeg": car_state_data.get("steeringAngleDeg", 0.0),
        "leftBlinker": car_state_data.get("leftBlinker", False),
        "rightBlinker": car_state_data.get("rightBlinker", False),
    }
    car_state_queue.append(car_state_entry)

def get_past_car_state_data(queue, step=0.5, steps=10):
    current_time = time.time()
    past_data = {
        "vEgos": [],
        "aEgos": [],
        "steeringAngleDegs": [],
        "leftBlinkers": [],
        "rightBlinkers": [],
    }

    if len(queue) == 0:
      for _ in range(steps):
        past_data["vEgos"].append(0.0)
        past_data["aEgos"].append(0.0)
        past_data["steeringAngleDegs"].append(0.0)
        past_data["leftBlinkers"].append(0)
        past_data["rightBlinkers"].append(0)
      return past_data

    for i in range(steps):
        target_time = current_time - (i * step)
        closest_entry = min(queue, key=lambda x: abs(x["timestamp"] - target_time))
        past_data["vEgos"].append(closest_entry["vEgo"])
        past_data["aEgos"].append(closest_entry["aEgo"])
        past_data["steeringAngleDegs"].append(closest_entry["steeringAngleDeg"])
        past_data["leftBlinkers"].append(1 if closest_entry["leftBlinker"] else 0)
        past_data["rightBlinkers"].append(1 if closest_entry["rightBlinker"] else 0)

    return past_data

def process_camera_frame(buf) -> np.ndarray:
    """
    VisionBufから実際の画像データを取得し、E2Eモデル用に前処理を実行
    """
    if buf is None:
        raise ValueError("VisionBuf is None")
    yuv_img = buf.data
    # YUV420をRGBに変換
    rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_I420)
    # 画像のリサイズ (元解像度 → 224x224)
    resized_img = cv2.resize(
        rgb_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    # [0, 255] → [0, 1] 正規化
    normalized_img = resized_img.astype(np.float32) / 255.0
    # HWC → CHW (Height, Width, Channel → Channel, Height, Width)
    # PyTorchモデルの入力形式に変換
    transposed_img = normalized_img.transpose(2, 0, 1)
    return transposed_img


class DecodedFrame:
    def __init__(self, data: np.ndarray, width: int, height: int, frame_id: int, timestamp_sof: int, timestamp_eof: int):
        self.data = data
        self.width = width
        self.height = height
        self.frame_id = frame_id
        self.timestamp_sof = timestamp_sof
        self.timestamp_eof = timestamp_eof


def _decode_ffvhuff_packet_in_memory(payload: bytes, width: int, height: int):
    output = io.BytesIO()
    out = av.open(output, mode="w", format="avi")
    stream = out.add_stream("ffvhuff", rate=20)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    packet = av.packet.Packet(payload)
    packet.pts = 0
    packet.dts = 0
    packet.stream = stream
    out.mux(packet)
    out.close()

    output.seek(0)
    inp = av.open(output, mode="r", format="avi")
    try:
        for demux_packet in inp.demux(video=0):
            frames = demux_packet.decode()
            if len(frames) > 0:
                return frames[-1]
    finally:
        inp.close()
    return None


class EncodeDataDecoder:
    def decode(self, msg) -> Optional[DecodedFrame]:
        if msg.idx.type != log.EncodeIndex.Type.bigBoxLossless:
            print(f"Unsupported EncodeIndex type: {msg.idx.type}")
            return None

        width = int(msg.width)
        height = int(msg.height)
        frame = _decode_ffvhuff_packet_in_memory(bytes(msg.data), width, height)

        yuv = np.ascontiguousarray(frame.to_ndarray(format="yuv420p"))
        return DecodedFrame(
            data=yuv,
            width=width,
            height=height,
            frame_id=int(msg.idx.frameId),
            timestamp_sof=int(msg.idx.timestampSof),
            timestamp_eof=int(msg.idx.timestampEof),
        )

class ScratchModelState:
    """
    スクラッチモデルの状態とデータを管理するクラス
    """
    session: ort.InferenceSession  # ONNXランタイムセッション
    inputs: Dict[str, np.ndarray]  # モデル入力データ
    output: Dict[str, float]  # モデル出力データ

    def __init__(self):
        """
        ScratchModelStateの初期化
        """
        self.session = ort.InferenceSession(
            E2E_MODEL_PATHS[ModelRunner.ONNX].as_posix(), providers=["CPUExecutionProvider"]
        )

        self.inputs = {
            "mainCamera": np.zeros((1, 3, 224, 224), dtype=np.float32),
            "zoomCamera": np.zeros((1, 3, 224, 224), dtype=np.float32),
            "navVector": np.zeros((1, 150), dtype=np.float32),
            "carState": np.zeros((1, CAR_STATE_DIM, PREDICTION_HORIZON), dtype=np.float32),  # 統合されたcarState
        }

        self.output = {
            "pred_vEgo": float(0.0),
            "pred_aEgo": float(0.0),
            "pred_steeringAngleDeg": float(0.0),
        }

    def run(
      self, buf, wbuf, inputs: Dict[str, np.ndarray]
    ) -> Optional[Dict[str, float|List[float]]]:
        """
        モデルの推論実行メイン関数
        """
        main_camera_input = process_camera_frame(buf)
        zoom_camera_input = process_camera_frame(wbuf)
        self.inputs["mainCamera"] = np.expand_dims(main_camera_input, axis=0)
        self.inputs["zoomCamera"] = np.expand_dims(zoom_camera_input, axis=0)

        past_car_state_data = get_past_car_state_data(car_state_queue, step=0.5, steps=PREDICTION_HORIZON)
        car_state_tensor = np.stack([
            np.array(past_car_state_data["vEgos"], dtype=np.float32) / 10,  # スケーリング
            np.array(past_car_state_data["aEgos"], dtype=np.float32),
            np.array(past_car_state_data["steeringAngleDegs"], dtype=np.float32) / 100,  # スケーリング
            np.array(past_car_state_data["leftBlinkers"], dtype=np.float32),
            np.array(past_car_state_data["rightBlinkers"], dtype=np.float32),
        ], axis=0)

        self.inputs["carState"] = np.expand_dims(car_state_tensor, axis=0)
        self.inputs["navVector"] = np.expand_dims(inputs.get(
            "navVector", np.zeros(150, dtype=np.float32)
        ), axis=0)

        pred_vEgos, pred_aEgos, pred_steeringAngleDegs = self.session.run(None, self.inputs)
        vEgos_plan: List[float] = (pred_vEgos[0] * 10.0).tolist()  # m/sにスケーリング
        self.output["pred_vEgo"] = vEgos_plan[0]
        self.output["pred_aEgo"] = float(pred_aEgos[0][0])
        self.output["pred_steeringAngleDeg"] = float(pred_steeringAngleDegs[0][0] * 100.0)  # degにスケーリング
        self.output["vEgos_plan"] = vEgos_plan
        return self.output

def main(addr: str):
    cloudlog.warning("modeld init")

    sentry.set_tag("daemon", PROCESS_NAME)
    cloudlog.bind(daemon=PROCESS_NAME)
    setproctitle(PROCESS_NAME)
    config_realtime_process(7, 54)

    try:
        e2e_model = ScratchModelState()
        cloudlog.warning("e2e models loaded, e2e modeld starting")
    except Exception as e:
        cloudlog.error(f"modeld failed to load models: {e}")
        raise

    print(f"Subscribing to carState, roadEncodeData, wideRoadEncodeData, modelV2 from {addr}")
    sm = SubMaster(["carState", "roadEncodeData", "wideRoadEncodeData", "modelV2"], addr=addr)
    pm = PubMaster(["E2EOutput"])

    road_decoder = EncodeDataDecoder()
    wide_decoder = EncodeDataDecoder()
    latest_main = None
    latest_extra = None

    counter = 0
    while True:
        time_s = time.time()
        counter += 1
        sm.update(1000)
        new_frame = False

        if sm.updated["wideRoadEncodeData"]:
            decoded = wide_decoder.decode(sm["wideRoadEncodeData"])
            if decoded is not None:
                latest_main = decoded
                new_frame = True

        if sm.updated["roadEncodeData"]:
            decoded = road_decoder.decode(sm["roadEncodeData"])
            if decoded is not None:
                latest_extra = decoded
                new_frame = True

        if sm.updated["modelV2"]:
            try:
                use_e2eoutput = int(sm["modelV2"].action.useE2eOutput)
                desired_curvature = float(sm["modelV2"].action.desiredCurvature)
                print(f"[scratchmodeld] modelV2.action.useE2eOutput={use_e2eoutput}, desiredCurvature={desired_curvature:.6f}")
            except AttributeError:
                print("[scratchmodeld] modelV2.action.useE2eOutput is unavailable. Rebuild cereal/generated bindings first.")

        if latest_main is None or latest_extra is None:
            if counter % 50 == 0:
                print("Waiting for roadEncodeData and wideRoadEncodeData ...")
            continue

        if not new_frame:
            continue

        buf_main = latest_main
        buf_extra = latest_extra

        if abs(buf_main.timestamp_sof - buf_extra.timestamp_sof) > 10000000:
            print(f"Frame timestamp mismatch: main {buf_main.timestamp_sof}, extra {buf_extra.timestamp_sof}")
            if buf_main.timestamp_sof > buf_extra.timestamp_sof:
                latest_extra = None
            else:
                latest_main = None
            continue

        car_state_msg = sm["carState"]
        if car_state_msg is not None:
            basic_attrs = [
                    "vEgo",
                    "aEgo",
                    "steeringAngleDeg",
                    "leftBlinker",
                    "rightBlinker",
            ]
            car_state_input = {}
            for attr in basic_attrs:
                if hasattr(car_state_msg, attr):
                    value = getattr(car_state_msg, attr)
                    car_state_input[attr] = value
                    print(f"{attr}: {value} (exists)")
                else:
                    print(f"{attr}: NOT FOUND")
            update_car_state_queue(car_state_input)
        else:
            print("carState message is None!")

        e2e_inputs: Dict[str, np.ndarray] = {
            "carState": sm["carState"],
        }
        e2e_model_output = e2e_model.run(buf_main, buf_extra, e2e_inputs)

        if e2e_model_output is None:
            print("E2E model output is None!")
            continue

        pred_vEgo = e2e_model_output["pred_vEgo"]
        pred_aEgo = e2e_model_output["pred_aEgo"]
        pred_steeringAngleDeg = e2e_model_output["pred_steeringAngleDeg"]
        pred_vEgos_plan = e2e_model_output["vEgos_plan"]

        e2e_output_send = messaging.new_message('e2eOutput')
        e2e_output_send.valid = True
        e2e_output_send.timestamp = int(time.time() * 1e9)
        e2e_output_send.E2EOutput.vEgo = float(pred_vEgo)
        e2e_output_send.E2EOutput.aEgo = float(pred_aEgo)
        e2e_output_send.E2EOutput.steeringAngleDeg = float(pred_steeringAngleDeg)
        e2e_output_send.E2EOutput.vEgosPlan = pred_vEgos_plan
        pm.send('e2eOutput', e2e_output_send)

        time_e = time.time()
        latency_ms = (time_e - time_s) * 1000

        cloudlog.debug(f"[E2E model predictions] steer: {pred_steeringAngleDeg:.6f} Deg, acc: {pred_aEgo:.6f} m/s², vel: {pred_vEgo:.6f} m/s, latency: {latency_ms:.2f} ms")
        print(f"pred_vEgo: {pred_vEgo:.6f} m/s, pred_aEgo: {pred_aEgo:.6f} m/s², pred_steeringAngleDeg: {pred_steeringAngleDeg:.6f} Deg, latency: {latency_ms:.2f} ms")


if __name__ == "__main__":
  try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", help="Address to subscribe to", default="127.0.0.1")
    args = parser.parse_args()
    main(args.addr)
  except KeyboardInterrupt:
    cloudlog.warning(f"child {PROCESS_NAME} got SIGINT")
  except Exception:
    sentry.capture_exception()
    raise

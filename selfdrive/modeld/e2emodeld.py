#!/usr/bin/env python3
"""
E2E (End-to-End) 自動運転モデル実行デーモン
"""

import os
import time
import numpy as np
# import cv2
from cereal import car
from pathlib import Path
from typing import Dict, List, Optional
from setproctitle import setproctitle
from cereal.messaging import PubMaster, SubMaster
# from cereal.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
import onnxruntime as ort
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process
from openpilot.selfdrive import sentry
from openpilot.selfdrive.car.car_helpers import get_demo_car_params
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.modeld.runners import ModelRunner
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.models.commonmodel_pyx import ModelFrame, CLContext
from collections import deque

# ===== プロセス設定 =====
PROCESS_NAME = "selfdrive.modeld.e2emodeld"
SEND_RAW_PRED = os.getenv("SEND_RAW_PRED")  # デバッグ用: 生の予測値送信フラグ
SEND_E2E_OUTPUT = os.getenv(
    "SEND_E2E_OUTPUT", "1"
)  # E2E出力を常に送信（デフォルト有効） 

# ===== モデルファイルパス設定 =====
# カスタム学習済みE2Eモデルのパス設定
MODEL_PATHS = {
    ModelRunner.ONNX: Path(__file__).parent
    # / "models/checkpoint_epoch_57_best.onnx"  # v2.1 Transformer
    # / "models/checkpoint_epoch_90_best.onnx"  # v2.1 LSTM
    / "models/v2.2_lstm.onnx"  # v2.2 LSTM

}

E2E_MODEL_FREQ = 10.0  # 10Hz
IMAGE_SIZE = 224

# 新しいcarStateの次元を定義
CAR_STATE_DIM = 5
PREDICTION_HORIZON = 10

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

    for i in range(steps):
        target_time = current_time - (i * step)
        closest_entry = min(queue, key=lambda x: abs(x["timestamp"] - target_time))
        past_data["vEgos"].append(closest_entry["vEgo"])
        past_data["aEgos"].append(closest_entry["aEgo"])
        past_data["steeringAngleDegs"].append(closest_entry["steeringAngleDeg"])
        past_data["leftBlinkers"].append(1 if closest_entry["leftBlinker"] else 0)
        past_data["rightBlinkers"].append(1 if closest_entry["rightBlinker"] else 0)

    return past_data


# def process_camera_frame(buf: VisionBuf) -> np.ndarray:
#     """
#     VisionBufから実際の画像データを取得し、E2Eモデル用に前処理を実行
#     """
#     try:
#         # VisionBufの有効性チェック
#         if buf is None:
#             cloudlog.warning("VisionBuf is None, using dummy image")
#             return np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

#         # YUV420: Y(輝度) + U/V(色差)が縦方向に1.5倍のサイズで格納
#         yuv_img = np.frombuffer(buf.data, dtype=np.uint8).reshape(
#             (buf.height + buf.height // 2, buf.width)
#         )

#         # YUV420をRGBに変換
#         rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_I420)

#         # 画像のリサイズ (元解像度 → 224x224)
#         resized_img = cv2.resize(
#             rgb_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
#         )

#         # [0, 255] → [0, 1] 正規化
#         normalized_img = resized_img.astype(np.float32) / 255.0

#         # HWC → CHW (Height, Width, Channel → Channel, Height, Width)
#         # PyTorchモデルの入力形式に変換
#         transposed_img = normalized_img.transpose(2, 0, 1)

#         return transposed_img

#     except Exception as e:
#         cloudlog.error(f"Error processing camera frame: {e}")
#         # エラー時はゼロで埋めたダミー画像を返す
#         return np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)


class FrameMeta:
    """
    カメラフレームのメタデータを管理するクラス

    フレームID、タイムスタンプなどの情報を格納し、
    フレーム同期やドロップフレーム検出に使用されます。
    """

    frame_id: int = 0  # フレーム通番
    timestamp_sof: int = 0  # フレーム開始時刻（nanosecond）
    timestamp_eof: int = 0  # フレーム終了時刻（nanosecond）

    def __init__(self, vipc=None):
        """
        VisionIpcClientから メタデータを初期化

        Args:
          vipc: VisionIpcClient - カメラクライアント（Noneの場合はデフォルト値を使用）
        """
        if vipc is not None:
            self.frame_id, self.timestamp_sof, self.timestamp_eof = (
                vipc.frame_id,
                vipc.timestamp_sof,
                vipc.timestamp_eof,
            )


# class E2EModelState:
#     """
#     E2E（End-to-End）モデルの状態とデータを管理するクラス
#     """

#     # クラス属性の型ヒント
#     frame: ModelFrame  # メインカメラフレーム処理用
#     wide_frame: ModelFrame  # ワイドカメラフレーム処理用
#     session: ort.InferenceSession  # ONNXランタイムセッション
#     inputs: Dict[str, np.ndarray]  # モデル入力データ
#     output: Dict[str, float]  # モデル出力データ

#     def __init__(self, context: CLContext):
#         """
#         E2EModelStateの初期化
#         """
#         # フレーム処理用オブジェクトの初期化
#         self.frame = ModelFrame(context)  # メインカメラ用
#         self.wide_frame = ModelFrame(context)  # ワイドカメラ用

#         self.session = ort.InferenceSession(
#             MODEL_PATHS[ModelRunner.ONNX].as_posix(), providers=["CPUExecutionProvider"]
#         )

#         self.inputs = {
#             "mainCamera": np.zeros((1, 3, 224, 224), dtype=np.float32),
#             "zoomCamera": np.zeros((1, 3, 224, 224), dtype=np.float32),
#             "navVector": np.zeros((1, 150), dtype=np.float32),
#             "carState": np.zeros((1, CAR_STATE_DIM, PREDICTION_HORIZON), dtype=np.float32),  # 統合されたcarState
#         }

#         self.output = {
#             "pred_vEgo": float(0.0),
#             "pred_aEgo": float(0.0),
#             "pred_steeringAngleDeg": float(0.0),
#         }

#     def run(
#         self, buf: VisionBuf, wbuf: VisionBuf, inputs: Dict[str, np.ndarray]
#     ) -> Optional[Dict[str, float|List[float]]]:
#         """
#         E2Eモデルの推論実行メイン関数
#         """

#         try:
#             main_camera_input = process_camera_frame(buf)
#             zoom_camera_input = process_camera_frame(wbuf)
#             self.inputs["mainCamera"] = np.expand_dims(main_camera_input, axis=0)
#             self.inputs["zoomCamera"] = np.expand_dims(zoom_camera_input, axis=0)
#         except Exception as e:
#             cloudlog.error(f"Error processing camera inputs: {e}")

#         try:
#             # 統合されたcarStateを作成
#             past_car_state_data = get_past_car_state_data(car_state_queue, step=0.5, steps=PREDICTION_HORIZON)
#             car_state_tensor = np.stack([
#                 np.array(past_car_state_data["vEgos"], dtype=np.float32) / 10,  # スケーリング
#                 np.array(past_car_state_data["aEgos"], dtype=np.float32),
#                 np.array(past_car_state_data["steeringAngleDegs"], dtype=np.float32) / 100,  # スケーリング
#                 np.array(past_car_state_data["leftBlinkers"], dtype=np.float32),
#                 np.array(past_car_state_data["rightBlinkers"], dtype=np.float32),
#             ], axis=0)  # (CAR_STATE_DIM, PREDICTION_HORIZON)

#             self.inputs["carState"] = np.expand_dims(car_state_tensor, axis=0)  # (1, CAR_STATE_DIM, PREDICTION_HORIZON)
#         except Exception as e:
#             cloudlog.error(f"Error processing carState input: {e}")

#         try:
#             self.inputs["navVector"] = np.expand_dims(inputs.get(
#                 "navVector", np.zeros(150, dtype=np.float32)
#             ), axis=0)
#         except Exception as e:
#             cloudlog.error(f"Error processing navVector input: {e}")

#         pred_vEgos, pred_aEgos, pred_steeringAngleDegs = self.session.run(None, self.inputs)
#         vEgos_plan: List[float] = (pred_vEgos[0] * 10.0).tolist()  # m/sにスケーリング
#         self.output["pred_vEgo"] = vEgos_plan[0]
#         self.output["pred_aEgo"] = float(pred_aEgos[0][0])
#         self.output["pred_steeringAngleDeg"] = float(pred_steeringAngleDegs[0][0] * 100.0)  # degにスケーリング
#         self.output["vEgos_plan"] = vEgos_plan
#         return self.output


def main(demo=False):
    """
    E2Eモデルデーモンのメイン関数
    """
    cloudlog.warning("e2emodeld init")

    # ===== プロセス設定の初期化 =====
    sentry.set_tag("daemon", PROCESS_NAME)  # Sentryエラー追跡用タグ設定
    cloudlog.bind(daemon=PROCESS_NAME)  # ログにプロセス名をバインド
    setproctitle(PROCESS_NAME)  # プロセス名を設定（psコマンドで確認可能）
    config_realtime_process(7, 54)  # リアルタイムプロセス設定（CPU7番、優先度54）

    # ===== OpenCLコンテキストとE2Eモデルの初期化 =====
    try:
        cloudlog.warning("setting up CL context")
        cl_context = CLContext()  # OpenCL実行コンテキスト（GPU処理用）
        cloudlog.warning("CL context ready; loading E2E model")
        #検証のためコメントアウト
        # model = E2EModelState(cl_context)  # E2Eモデルの初期化
        cloudlog.warning("E2E model loaded, e2emodeld starting")
    except Exception as e:
        cloudlog.error(f"Failed to initialize E2E model: {e}")
        import traceback

        cloudlog.error(
            f"E2E model initialization error traceback: {traceback.format_exc()}"
        )
        raise

    # ===== カメラクライアントの設定（シミュレータ・実機対応） =====
    # try:
    #     cloudlog.warning("Setting up vision clients...")

    #     # カメラストリームの自動検出（環境に応じて適応）
    #     timeout_count = 0
    #     max_timeout = 50  # 5秒間の試行

    #     while True:
    #         available_streams = VisionIpcClient.available_streams(
    #             "camerad", block=False
    #         )
    #         if available_streams:
    #             use_extra_client = (
    #                 VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams
    #                 and VisionStreamType.VISION_STREAM_ROAD in available_streams
    #             )
    #             main_wide_camera = (
    #                 VisionStreamType.VISION_STREAM_ROAD not in available_streams
    #             )
    #             break

    #         timeout_count += 1
    #         if timeout_count >= max_timeout:
    #             cloudlog.error("⚠️ Timeout waiting for camera streams")
    #             if demo:
    #                 cloudlog.warning(
    #                     "🎮 Demo mode: Proceeding without camera streams (may use dummy data)"
    #                 )
    #                 # デモモードでは続行を許可
    #                 available_streams = []
    #                 use_extra_client = False
    #                 main_wide_camera = True
    #                 break
    #             else:
    #                 raise RuntimeError("Camera streams not available in real mode")

    #         time.sleep(0.1)

    #     vipc_client_main_stream = (
    #         VisionStreamType.VISION_STREAM_WIDE_ROAD
    #         if main_wide_camera
    #         else VisionStreamType.VISION_STREAM_ROAD
    #     )
    #     vipc_client_main = VisionIpcClient(
    #         "camerad", vipc_client_main_stream, True, cl_context
    #     )
    #     vipc_client_extra = VisionIpcClient(
    #         "camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, cl_context
    #     )
    #     cloudlog.warning(
    #         f"📷 Vision config: main_wide_camera={main_wide_camera}, use_extra_client={use_extra_client}"
    #     )

    #     while not vipc_client_main.connect(False):
    #         time.sleep(0.1)
    #     while not vipc_client_extra.connect(False):
    #         time.sleep(0.1)

    #     # 接続成功の確認
    #     if vipc_client_main.connect(False):
    #         cloudlog.warning(
    #             f"✅ Main camera connected: {vipc_client_main.buffer_len} buffers "
    #         )
    #     if use_extra_client and vipc_client_extra.connect(False):
    #         cloudlog.warning(
    #             f"✅ Extra camera connected: {vipc_client_extra.buffer_len} buffers "
    #         )

    # except Exception as e:
    #     cloudlog.error(f"Failed to setup vision clients: {e}")
    #     if demo:
    #         cloudlog.warning("🎮 Demo mode: Continuing despite vision setup failure")
    #         vipc_client_main = None
    #         vipc_client_extra = None
    #         use_extra_client = False
    #         main_wide_camera = True
    #     else:
    #         raise

    # try:
    #     pm = PubMaster(["e2eOutput"])
    #     sm = SubMaster(["carState", "navInstruction"])
    # except Exception as e:
    #     cloudlog.error(f"Failed to setup messaging: {e}")
    #     raise


    if demo:
        CP = get_demo_car_param
    # params = Params()

    # # setup filter to track dropped frames
    # frame_dropped_filter = FirstOrderFilter(0.0, 10.0, 1.0 / ModelConstants.MODEL_FREQ)
    # last_vipc_frame_id = 0
    # run_count = 0
    # live_calib_seen = False
    # nav_instructions = np.zeros(ModelConstants.NAV_INSTRUCTION_LEN, dtype=np.float32)
    # buf_main, buf_extra = None, None
    # meta_main = FrameMeta()
    # meta_extra = FrameMeta()s()
    # else:
    #     with car.CarParams.from_bytes(params.get("CarParams", block=True)) as msg:
    #         CP = msg
    # cloudlog.info("e2emodeld got CarParams: %s", CP.carName)

    # DH = DesireHelper()

    cloudlog.warning("E2E model main loop starting")

    # E2E専用の更新頻度制御とフレーム処理
    last_e2e_update_time = 0.0
    e2e_update_interval = 1.0 / E2E_MODEL_FREQ  # 10Hz間隔
    loop_count = 0  # ループカウンター追加

    while True:
        current_time = time.monotonic()
        loop_count += 1

        # if loop_count % 100 == 1:
        #     with open("/tmp/e2e_car_state_debug.log", "a") as f:
        #         import time as time_module
        #
        #         f.write(f"{time_module.time()}: Main loop iteration {loop_count}\n")
        #         f.flush()

        # E2E更新頻度制御
        if current_time - last_e2e_update_time < e2e_update_interval:
            # time.sleep(0.001)
            time.sleep(0.01)
            continue

        # ===== カメラフレーム取得 =====
        # try:
        #     # メインカメラフレーム処理
        #     cloudlog.debug(
        #         f"Attempting to receive main frame, meta_main.timestamp: {meta_main.timestamp_sof}, meta_extra.timestamp: {meta_extra.timestamp_sof}"
        #     )

        #     # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
        #     recv_attempts = 0
        #     max_attempts = 10  # 試行回数を増加

        #     while (
        #         meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000
        #         and recv_attempts < max_attempts
        #     ):
        #         buf_main = vipc_client_main.recv()
        #         meta_main = FrameMeta(vipc_client_main)
        #         recv_attempts += 1
        #         if buf_main is None:
        #             time.sleep(0.02)
        #             continue
        #         else:
        #             break

        #     # 追加カメラフレーム処理
        #     if use_extra_client:

        #         # Keep receiving extra frames until frame id matches main camera
        #         extra_recv_attempts = 0
        #         max_extra_attempts = 3
        #         while extra_recv_attempts < max_extra_attempts:
        #             buf_extra = vipc_client_extra.recv()
        #             meta_extra = FrameMeta(vipc_client_extra)
        #             extra_recv_attempts += 1
        #             if (
        #                 buf_extra is None
        #                 or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000
        #             ):
        #                 break

        #         # フレーム同期チェック
        #         if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        #             cloudlog.warning(
        #                 "frames out of sync! main: {} ({:.5f}), extra: {} ({:.5f})".format(
        #                     meta_main.frame_id,
        #                     meta_main.timestamp_sof / 1e9,
        #                     meta_extra.frame_id,
        #                     meta_extra.timestamp_sof / 1e9,
        #                 )
        #             )

        #     else:
        #         # シングルカメラモード
        #         buf_extra = buf_main
        #         meta_extra = meta_main

        # except Exception as e:
        #     cloudlog.error(f"Camera frame processing error: {e}")

        # sm.update(0)

        # try:
        #     car_state_msg = sm["carState"]
        #     car_state_valid = sm.valid["carState"]
        #     car_state_updated = sm.updated["carState"]

        #     if car_state_msg is not None:
        #         basic_attrs = [
        #             "vEgo",
        #             "aEgo",
        #             "steeringAngleDeg",
        #             "leftBlinker",
        #             "rightBlinker",
        #         ]
        #         car_state_input = {}
        #         for attr in basic_attrs:
        #             if hasattr(car_state_msg, attr):
        #                 value = getattr(car_state_msg, attr)
        #                 car_state_input[attr] = value
        #                 print(f"   {attr}: {value} (exists)", flush=True)
        #             else:
        #                 print(f"   {attr}: NOT FOUND", flush=True)
        #         update_car_state_queue(car_state_input)
        #     else:
        #         print("❌ carState message is None!", flush=True)

            # ファイルログにも記録
            # with open("/tmp/e2e_carstate_message_debug.log", "a") as f:
            #     import time as time_module

            #     f.write(
            #         f"{time_module.time()}: CarState Message - Valid:{car_state_valid}, Updated:{car_state_updated}, Type:{type(car_state_msg).__name__}\n"
            #     )
            #     f.flush()

        # except Exception as debug_error:
        #     print(f"❌ CAR STATE MESSAGE DEBUG ERROR: {debug_error}", flush=True)

        # desire = DH.desire
        # is_rhd = True  # 右ハンドル車として設定

        # if not live_calib_seen:
        #     live_calib_seen = True

        # traffic_convention = np.zeros(2)
        # traffic_convention[int(is_rhd)] = 1

        # vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
        # if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
        #     vec_desire[desire] = 1

        # nav_valid = sm.valid["navInstruction"]
        # nav_enabled = nav_valid

        # if not nav_enabled:
        #     nav_instructions[:] = 0

        # if nav_enabled and sm.updated["navInstruction"]:
        #     nav_instructions[:] = 0
        #     maneuver_processed = 0
        #     for maneuver in sm["navInstruction"].allManeuvers:
        #         distance_idx = 25 + int(maneuver.distance / 20)
        #         direction_idx = 0
        #         if maneuver.modifier in ("left", "slight left", "sharp left"):
        #             direction_idx = 1
        #         if maneuver.modifier in ("right", "slight right", "sharp right"):
        #             direction_idx = 2
        #         if 0 <= distance_idx < 50:
        #             final_idx = distance_idx * 3 + direction_idx
        #             nav_instructions[final_idx] = 1
        #             maneuver_processed += 1

        # # tracked dropped frames
        # vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
        # if run_count < 10:  # let frame drops warm up
        #     frame_dropped_filter.x = 0.0
        # run_count = run_count + 1

        # prepare_only = vipc_dropped_frames > 0
        # if prepare_only:
        #     cloudlog.error(
        #         f"skipping E2E model eval. Dropped {vipc_dropped_frames} frames"
        #     )

        # inputs: Dict[str, np.ndarray] = {
        #     "carState": sm["carState"],
        #     "nav_instructions": nav_instructions,
        # }

        # mt1 = time.perf_counter()
        # model_output = model.run(buf_main, buf_extra, inputs)
        # mt2 = time.perf_counter()
        # model_execution_time = mt2 - mt1

        # if model_output is not None:
        #     cloudlog.debug(f"E2E model execution time: {model_execution_time:.4f}s")

        #     try:
        #         pred_vEgo = model_output["pred_vEgo"]
        #         pred_aEgo = model_output["pred_aEgo"]
        #         pred_steeringAngleDeg = model_output["pred_steeringAngleDeg"]
        #         vEgos_plan = model_output.get("vEgos_plan", [])

        #         cloudlog.debug("E2E ONNX model predictions")
        #         cloudlog.debug(f"steer: {pred_steeringAngleDeg:.6f} Deg")
        #         cloudlog.debug(f"acc: {pred_aEgo:.6f} m/s², vel: {pred_vEgo:.6f} m/s")
        #         cloudlog.debug(f"execTime: {model_execution_time:.3f}ms")

        #         # 詳細デバッグ: モデル出力値をファイルにも記録
        #         debug_msg = f"steer={pred_steeringAngleDeg:.6f}, acc={pred_aEgo:.6f}, vel={pred_vEgo:.6f}, execTime={model_execution_time:.3f}ms"
        #         import time as time_module

        #         with open("/tmp/e2e_model_output_debug.log", "a") as f:
        #             f.write(f"{time_module.time()}: {debug_msg}\n")

        #         # rlogに記録するためのメッセージ送信
        #         if pm is not None:
        #             import cereal.messaging as messaging

        #             # e2eOutputメッセージ
        #             e2e_out_msg = messaging.new_message("e2eOutput")
        #             e2e_out_msg.e2eOutput.aEgo = pred_aEgo
        #             e2e_out_msg.e2eOutput.vEgo = pred_vEgo
        #             e2e_out_msg.e2eOutput.steeringAngleDeg = pred_steeringAngleDeg
        #             e2e_out_msg.e2eOutput.vEgoPlans = vEgos_plan
        #             e2e_out_msg.e2eOutput.timestamp = int(time.time_ns())
        #             e2e_out_msg.e2eOutput.isValid = True
        #             # 重要: メッセージレベルのvalidフラグも設定
        #             e2e_out_msg.valid = True
        #             pm.send("e2eOutput", e2e_out_msg)

        #         # E2E更新時間を記録
        #         last_e2e_update_time = current_time

        #     except Exception as e:
        #         cloudlog.error(f"Error processing E2E model output: {e}")
        #         import traceback

        #         cloudlog.error(f"E2E error traceback: {traceback.format_exc()}")

        #         # エラー時でも無効なメッセージを送信
        #         try:
        #             e2e_out_msg = messaging.new_message("e2eOutput")
        #             e2e_out_msg.e2eOutput.aEgo = 0.0
        #             e2e_out_msg.e2eOutput.vEgo = 0.0
        #             e2e_out_msg.e2eOutput.steeringAngleDeg = 0.0
        #             e2e_out_msg.e2eOutput.vEgoPlans = [0.0] * 10
        #             e2e_out_msg.e2eOutput.timestamp = int(time.time_ns())
        #             e2e_out_msg.e2eOutput.isValid = False
        #             # エラー時: メッセージレベルのvalidも無効に設定
        #             e2e_out_msg.valid = False
        #             pm.send("e2eOutput", e2e_out_msg)
        #             cloudlog.debug("E2E error: sent invalid message")
        #         except Exception as msg_error:
        #             cloudlog.error(f"Failed to send error message: {msg_error}")

        # # フレームIDの更新（次回のフレームドロップ検出用）
        # last_vipc_frame_id = meta_main.frame_id


# ===== メイン実行部 =====
if __name__ == "__main__":
    """
    E2Eモデルデーモンのエントリーポイント

    コマンドライン引数:
      --demo: デモモード（CarParamsを自動生成、実車なしでテスト可能）

    実行例:
      python selfdrive/modeld/e2emodeld.py          # 通常モード
      python selfdrive/modeld/e2emodeld.py --demo   # デモモード
    """
    try:
        # コマンドライン引数の解析
        import argparse

        parser = argparse.ArgumentParser(description="E2E自動運転モデル実行デーモン")
        parser.add_argument(
            "--demo",
            action="store_true",
            help="デモモード（CarParams自動生成、実車接続不要）",
        )
        args = parser.parse_args()

        # メイン関数の実行
        main(demo=args.demo)

    except KeyboardInterrupt:
        # Ctrl+Cによる正常終了
        cloudlog.warning(f"child {PROCESS_NAME} got SIGINT")
    except Exception:
        # 予期しないエラーの場合、Sentryに送信して再発生
        sentry.capture_exception()
        raise

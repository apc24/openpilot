#!/usr/bin/env python3
"""
E2E (End-to-End) 自動運転モデル実行デーモン

このファイルは、学習済みE2EモデルをOpenPilotの実行環境で動作させるためのモジュールです。

主な機能:
- カスタム学習済みONNXモデルの推論実行
- カメラ画像、車両状態、ナビゲーション情報の前処理
- E2E制御信号（加速度、ステアリング）の出力
- OpenPilotメッセージングシステムとの統合

使用方法:
python selfdrive/modeld/e2emodeld.py [--demo]

環境変数:
- SEND_RAW_PRED: 生の予測値を送信するかどうか
- SEND_E2E_OUTPUT: E2E出力を送信するかどうか（デフォルト: 1）
"""

import os
import time
import pickle
import numpy as np
import cv2
import cereal.messaging as messaging
from cereal import car, log
from pathlib import Path
from typing import Dict, Optional
from setproctitle import setproctitle
from cereal.messaging import PubMaster, SubMaster
from cereal.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive import sentry
from openpilot.selfdrive.car.car_helpers import get_demo_car_params
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.modeld.runners import ModelRunner, Runtime
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.models.commonmodel_pyx import ModelFrame, CLContext

# ===== プロセス設定 =====
PROCESS_NAME = "selfdrive.modeld.e2emodeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')                    # デバッグ用: 生の予測値送信フラグ
SEND_E2E_OUTPUT = os.getenv('SEND_E2E_OUTPUT', '1')          # E2E出力を常に送信（デフォルト有効）

# ===== モデルファイルパス設定 =====
# カスタム学習済みE2Eモデルのパス設定（epoch 19 最新版）
MODEL_PATHS = {
  ModelRunner.THNEED: Path(__file__).parent / 'models/checkpoint_epoch_19_best.thneed',  # GPU最適化版（利用可能な場合）
  ModelRunner.ONNX: Path(__file__).parent / 'models/checkpoint_epoch_19_best.onnx'       # 標準ONNX版
}

# モデルメタデータファイル（入力/出力形状情報を含む）
METADATA_PATH = Path(__file__).parent / 'models/supercombo_metadata.pkl'

# ===== モデル入力パラメータのデフォルト設定 =====
DEFAULT_CAR_STATE_DIM = 8      # 車両状態ベクターの次元数（速度、加速度、角度など）
DEFAULT_NAV_VECTOR_DIM = 150   # ナビゲーションベクターの次元数
DEFAULT_IMAGE_SIZE = 224       # 入力画像サイズ（224x224ピクセル）

# ===== E2E処理頻度設定 =====
E2E_MODEL_FREQ = 10.0          # E2Eモデル実行頻度: 10Hz（負荷軽減のため）

# ===== 画像前処理設定（ImageNet標準に準拠）=====
IMAGE_SIZE = 224                                                    # 入力画像サイズ
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet平均値（RGB順）
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # ImageNet標準偏差（RGB順）

def load_model_config() -> Dict[str, int]:
  """
  モデル設定ファイルから入力次元を読み込み
  
  メタデータファイルが存在する場合は、そこから実際のモデル入力次元を読み込みます。
  ファイルが存在しない場合は、デフォルト値を使用します。
  
  Returns:
    Dict[str, int]: モデル入力次元の辞書
      - car_state_dim: 車両状態ベクターの次元数
      - nav_vector_dim: ナビゲーションベクターの次元数  
      - image_size: 入力画像のサイズ（正方形）
  """
  # デフォルト設定で初期化
  config = {
    'car_state_dim': DEFAULT_CAR_STATE_DIM,
    'nav_vector_dim': DEFAULT_NAV_VECTOR_DIM,
    'image_size': DEFAULT_IMAGE_SIZE
  }
  
  try:
    # メタデータファイルから設定を読み込み
    if METADATA_PATH.exists():
      with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
        
      # メタデータに入力次元情報があれば更新
      if 'input_shapes' in metadata:
        input_shapes = metadata['input_shapes']
        # 各入力の最後の次元を取得（バッチ次元を除く）
        if 'carState' in input_shapes:
          config['car_state_dim'] = input_shapes['carState'][-1]  # 最後の次元
        if 'navVector' in input_shapes:
          config['nav_vector_dim'] = input_shapes['navVector'][-1]  # 最後の次元
        if 'mainCamera' in input_shapes:
          config['image_size'] = input_shapes['mainCamera'][-1]  # H=W（正方形画像）
          
      cloudlog.info(f"Model config loaded from metadata: {config}")
    else:
      cloudlog.warning(f"Metadata file not found, using defaults: {config}")
      
  except Exception as e:
    cloudlog.error(f"Error loading model config: {e}, using defaults")
    
  return config

# E2E専用の更新頻度（負荷バランスのため10Hzに戻す）
E2E_MODEL_FREQ = 10.0  # 10Hz（負荷軽減のため）

# 画像前処理の設定
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def process_camera_frame(buf: VisionBuf, transform_matrix: np.ndarray) -> np.ndarray:
  """
  VisionBufから実際の画像データを取得し、E2Eモデル用に前処理を実行
  
  前処理ステップ:
  1. YUV420フォーマットからRGBへ変換
  2. 224x224ピクセルにリサイズ
  3. [0,255] → [0,1] 正規化
  4. ImageNet標準正規化（平均値減算、標準偏差除算）
  5. HWC → CHW形式変換（PyTorchモデル互換）
  6. バッチ次元追加
  
  Args:
    buf: VisionBuf - カメラフレームデータ（YUV420形式）
    transform_matrix: np.ndarray - 変換行列（現在未使用）
    
  Returns:
    np.ndarray: 前処理済み画像テンソル (1, 3, 224, 224)
      - shape: (batch_size=1, channels=3, height=224, width=224)
      - dtype: float32
      - range: ImageNet正規化後の値域
  """
  try:
    # VisionBufの有効性チェック
    if buf is None:
      cloudlog.warning("VisionBuf is None, using dummy image")
      return np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    
    # Step 1: VisionBufの画像データを取得 (YUV420形式)
    # YUV420: Y(輝度) + U/V(色差)が縦方向に1.5倍のサイズで格納
    print("buf.width =", buf.width)
    print("buf.height =", buf.height)
    print("len(buf.data) =", len(buf.data))
    yuv_img = np.frombuffer(buf.data, dtype=np.uint8).reshape((buf.height + buf.height//2, buf.width))
    
    # Step 2: YUV420をRGBに変換（OpenCVを使用）
    rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_I420)
    
    # Step 3: 画像のリサイズ (元解像度 → 224x224)
    resized_img = cv2.resize(rgb_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # Step 4: [0, 255] → [0, 1] 正規化
    normalized_img = resized_img.astype(np.float32) / 255.0
    
    # Step 5: ImageNet標準正規化 (training.pyと同じ前処理)
    # 各チャンネル（R,G,B）に対して (pixel - mean) / std
    for i in range(3):
      normalized_img[:, :, i] = (normalized_img[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
    
    # Step 6: HWC → CHW (Height, Width, Channel → Channel, Height, Width)
    # PyTorchモデルの入力形式に変換
    chw_img = np.transpose(normalized_img, (2, 0, 1))
    
    # Step 7: バッチ次元を追加 (C, H, W) → (1, C, H, W)
    batch_img = chw_img[np.newaxis, :]
    
    # デバッグログ: 前処理結果の概要を出力
    cloudlog.debug(f"Camera frame processed: {buf.width}x{buf.height} → {IMAGE_SIZE}x{IMAGE_SIZE}, range: [{batch_img.min():.3f}, {batch_img.max():.3f}]")
    return batch_img
    
  except Exception as e:
    cloudlog.error(f"Error processing camera frame: {e}")
    # エラー時はゼロで埋めたダミー画像を返す
    return np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

def process_car_state(car_state_data: Dict, target_dim: int = 8) -> np.ndarray:
  """
  車両状態データを前処理してE2Eモデル入力形式に変換
  
  車両状態には以下の情報が含まれます:
  - 速度 (vEgo): m/s
  - 加速度 (aEgo): m/s²  
  - ステアリング角度 (steeringAngleDeg): 度
  - ヨー角速度 (yawRate): rad/s
  - ウインカー状態 (leftBlinker, rightBlinker): boolean
  - ペダル状態 (brakePressed, gasPressed): boolean
  
  Args:
    car_state_data: Dict - carStateメッセージデータ
    target_dim: int - 出力次元数（モデルの期待する入力次元）
    
  Returns:
    np.ndarray: 前処理済み車両状態ベクター (1, target_dim)
      - 正規化済み（主要な値は[-1, 1]範囲）
      - 不足する次元はゼロで埋める
  """
  try:
    # 指定された次元数でゼロ初期化
    car_state_input = np.zeros(target_dim, dtype=np.float32)
    
    if car_state_data is not None:
      # 基本的な車両状態パラメータを抽出・配列化
      state_values = [
        getattr(car_state_data, 'vEgo', 0.0),           # 0: 速度 (m/s)
        getattr(car_state_data, 'aEgo', 0.0),           # 1: 加速度 (m/s²)
        getattr(car_state_data, 'steeringAngleDeg', 0.0), # 2: ステアリング角度 (deg)
        getattr(car_state_data, 'yawRate', 0.0),        # 3: ヨー角速度 (rad/s)
        getattr(car_state_data, 'leftBlinker', 0.0),    # 4: 左ウインカー (0/1)
        getattr(car_state_data, 'rightBlinker', 0.0),   # 5: 右ウインカー (0/1)
        getattr(car_state_data, 'brakePressed', 0.0),   # 6: ブレーキ (0/1)
        getattr(car_state_data, 'gasPressed', 0.0),     # 7: アクセル (0/1)
      ]
      
      # target_dimまでのデータを設定（余分なデータは切り捨て）
      actual_len = min(len(state_values), target_dim)
      car_state_input[:actual_len] = state_values[:actual_len]
      
      # 主要パラメータの正規化（学習時と同じスケールに合わせる）
      if target_dim > 0:
        car_state_input[0] = np.clip(car_state_input[0] / 50.0, -1.0, 1.0)  # 速度: 50m/s基準で正規化
      if target_dim > 1:
        car_state_input[1] = np.clip(car_state_input[1] / 5.0, -1.0, 1.0)   # 加速度: 5m/s²基準で正規化
      if target_dim > 2:
        car_state_input[2] = np.clip(car_state_input[2] / 180.0, -1.0, 1.0) # 角度: 180度基準で正規化
    
    # バッチ次元を追加して返す
    return car_state_input.reshape(1, target_dim)
    
  except Exception as e:
    cloudlog.error(f"Error processing car state: {e}")
    # エラー時はゼロベクターを返す
    return np.zeros((1, target_dim), dtype=np.float32)

def process_nav_vector(nav_features: np.ndarray, target_dim: int = 150) -> np.ndarray:
  """
  ナビゲーションベクターを前処理してE2Eモデル入力形式に変換
  
  ナビゲーション情報には以下が含まれる可能性があります:
  - 目的地までの距離と方向
  - 道路種別（高速道路、一般道など）
  - 交通規制情報
  - ルート予測データ
  - 地図特徴量
  
  Args:
    nav_features: np.ndarray - ナビゲーション特徴量
    target_dim: int - 出力次元数（モデルの期待する入力次元）
    
  Returns:
    np.ndarray: 前処理済みナビベクター (1, target_dim)
      - 値は[-1, 1]範囲にクリップ
      - 不足する次元はゼロで埋める
  """
  try:
    # 指定された次元数でゼロ初期化
    nav_vector_input = np.zeros(target_dim, dtype=np.float32)
    
    if nav_features is not None and len(nav_features) > 0:
      # 入力特徴量を指定次元数まで設定（余分なデータは切り捨て）
      nav_len = min(len(nav_features), target_dim)
      nav_vector_input[:nav_len] = nav_features[:nav_len]
      
      # 値の範囲を制限（異常値対策と学習時の前処理との整合性）
      nav_vector_input = np.clip(nav_vector_input, -1.0, 1.0)
    
    # バッチ次元を追加して返す
    return nav_vector_input.reshape(1, target_dim)
    
  except Exception as e:
    cloudlog.error(f"Error processing nav vector: {e}")
    # エラー時はゼロベクターを返す
    return np.zeros((1, target_dim), dtype=np.float32)

class FrameMeta:
  """
  カメラフレームのメタデータを管理するクラス
  
  フレームID、タイムスタンプなどの情報を格納し、
  フレーム同期やドロップフレーム検出に使用されます。
  """
  frame_id: int = 0          # フレーム通番
  timestamp_sof: int = 0     # フレーム開始時刻（nanosecond）
  timestamp_eof: int = 0     # フレーム終了時刻（nanosecond）

  def __init__(self, vipc=None):
    """
    VisionIpcClientから メタデータを初期化
    
    Args:
      vipc: VisionIpcClient - カメラクライアント（Noneの場合はデフォルト値を使用）
    """
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof

class E2EModelState:
  """
  E2E（End-to-End）モデルの状態とデータを管理するクラス
  
  主な機能:
  - 学習済みONNXモデルの読み込みと初期化
  - カメラフレーム、車両状態、ナビゲーション情報の管理
  - モデル推論の実行と結果の解析
  - OpenPilotメッセージング形式での出力
  """
  # クラス属性の型ヒント
  frame: ModelFrame                     # メインカメラフレーム処理用
  wide_frame: ModelFrame               # ワイドカメラフレーム処理用  
  inputs: Dict[str, np.ndarray]        # モデル入力データのバッファ
  output: np.ndarray                   # モデル出力データのバッファ
  prev_desire: np.ndarray              # 前回のdesire状態（変化検出用）
  model: ModelRunner                   # ONNXモデル実行エンジン
  model_config: Dict[str, int]         # モデル設定（入力次元など）

  def __init__(self, context: CLContext):
    """
    E2EModelStateの初期化
    
    Args:
      context: CLContext - OpenCL実行コンテキスト（GPU処理用）
    """
    # モデル設定を読み込み（メタデータファイルまたはデフォルト値）
    self.model_config = load_model_config()
    cloudlog.info(f"E2E Model initialized with config: {self.model_config}")
    
    # フレーム処理用オブジェクトの初期化
    self.frame = ModelFrame(context)        # メインカメラ用
    self.wide_frame = ModelFrame(context)   # ワイドカメラ用
    
    # desire状態追跡用（前回状態との比較に使用）
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    
    # 標準OpenPilotモデル用の入力バッファ初期化（互換性維持のため）
    self.inputs = {
      'desire': np.zeros(ModelConstants.DESIRE_LEN * (ModelConstants.HISTORY_BUFFER_LEN+1), dtype=np.float32),
      'traffic_convention': np.zeros(ModelConstants.TRAFFIC_CONVENTION_LEN, dtype=np.float32),
      'lateral_control_params': np.zeros(ModelConstants.LATERAL_CONTROL_PARAMS_LEN, dtype=np.float32),
      'prev_desired_curv': np.zeros(ModelConstants.PREV_DESIRED_CURV_LEN * (ModelConstants.HISTORY_BUFFER_LEN+1), dtype=np.float32),
      'nav_features': np.zeros(ModelConstants.NAV_FEATURE_LEN, dtype=np.float32),
      'nav_instructions': np.zeros(ModelConstants.NAV_INSTRUCTION_LEN, dtype=np.float32),
      'features_buffer': np.zeros(ModelConstants.HISTORY_BUFFER_LEN * ModelConstants.FEATURE_LEN, dtype=np.float32),
    }

    # ===== モデルメタデータとアウトプット設定の初期化 =====
    # メタデータファイルの存在確認（モデル出力形状情報を含む）
    if METADATA_PATH.exists():
      # メタデータから出力サイズと構造を読み込み
      with open(METADATA_PATH, 'rb') as f:
        model_metadata = pickle.load(f)
      self.output_slices = model_metadata['output_slices']  # 出力データの分割方法
      net_output_size = model_metadata['output_shapes']['outputs'][1]  # 出力ベクターサイズ
    else:
      cloudlog.warning("supercombo_metadata.pkl not found, using default values for custom E2E model")
      # カスタムONNXモデル用のデフォルト設定
      self.output_slices = {}                                        # 出力分割なし
      net_output_size = 2                                           # control_output: [aEgo, steeringTorque]

    # モデル出力バッファの初期化
    self.output = np.zeros(net_output_size, dtype=np.float32)
    self.parser = Parser()  # 出力パーサー（標準OpenPilotモデル用、互換性維持）

    # ===== E2E ONNXモデルの読み込みと初期化 =====
    cloudlog.warning(f"Loading E2E model from {MODEL_PATHS[ModelRunner.ONNX]}")
    
    # カスタムONNXモデルの出力バッファを設定（固定サイズ: 2要素）
    custom_output_buffer = np.zeros(2, dtype=np.float32)  # [aEgo, steeringTorque]
    
    # ModelRunnerの初期化（GPU実行、プリプロセス無効）
    self.model = ModelRunner(MODEL_PATHS, custom_output_buffer, Runtime.GPU, False, context)
    
    # ===== カスタムONNXモデルの入力定義 =====
    # 動的な次元数でモデル入力を設定（メタデータまたはデフォルト値を使用）
    self.model.addInput("carState", None)        # [batch, car_state_dim] - 車両状態ベクター
    self.model.addInput("mainCamera", None)      # [batch, 3, image_size, image_size] - メインカメラ画像
    self.model.addInput("zoomCamera", None)      # [batch, 3, image_size, image_size] - ズームカメラ画像
    self.model.addInput("navVector", None)       # [batch, nav_vector_dim] - ナビゲーションベクター

  def slice_outputs(self, model_outputs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    モデル出力を解析して辞書形式に変換
    
    カスタムE2Eモデルの出力形式:
    - model_outputs[0]: aEgo (加速度指令) [m/s²]
    - model_outputs[1]: steeringTorque (ステアリングトルク指令) [Nm]
    
    Args:
      model_outputs: np.ndarray - モデルの生出力 [2] 
      
    Returns:
      Dict[str, np.ndarray]: 解析済み出力辞書
        - 'control_output': 制御信号 [1, 2]
        - 'raw_pred': 生の予測値（デバッグ用、SEND_RAW_PRED=1の場合）
    """
    # E2Eモデル出力の解析: [aEgo, steeringTorque]
    if len(model_outputs) >= 2:
      parsed_model_outputs = {
        'control_output': model_outputs.reshape(1, -1)  # バッチ形式に変換: [1, 2]
      }
    else:
      # 出力サイズが不正な場合のフォールバック処理
      cloudlog.warning(f"E2E model output size insufficient: {len(model_outputs)} < 2")
      parsed_model_outputs = {'outputs': model_outputs[np.newaxis, :]}
    
    # デバッグ用: 生の予測値も含める（環境変数で制御）
    if SEND_RAW_PRED:
      parsed_model_outputs['raw_pred'] = model_outputs.copy()
      
    return parsed_model_outputs

  def run(self, buf: VisionBuf, wbuf: VisionBuf, transform: np.ndarray, transform_wide: np.ndarray,
                inputs: Dict[str, np.ndarray], prepare_only: bool) -> Optional[Dict[str, np.ndarray]]:
    """
    E2Eモデルの推論実行メイン関数
    
    処理フロー:
    1. カメラ画像の前処理（YUV→RGB、リサイズ、正規化）
    2. 車両状態の前処理（正規化、次元調整）
    3. ナビゲーション情報の前処理
    4. モデル推論の実行
    5. 結果の返却
    
    Args:
      buf: VisionBuf - メインカメラフレーム
      wbuf: VisionBuf - ワイドカメラフレーム  
      transform: np.ndarray - メインカメラ変換行列
      transform_wide: np.ndarray - ワイドカメラ変換行列
      inputs: Dict[str, np.ndarray] - その他の入力データ
      prepare_only: bool - 準備のみでモデル実行しない場合True
      
    Returns:
      Optional[Dict[str, np.ndarray]]: モデル出力（prepare_only=Trueの場合はNone）
    """
    # モデル設定から動的に画像サイズを取得
    image_size = self.model_config['image_size']
    
    # ===== Step 1: カメラ画像の前処理 =====
    try:
      # メインカメラとズームカメラの画像を実際に処理（ダミーではなく実画像）
      main_camera_input = process_camera_frame(buf, transform)
      zoom_camera_input = process_camera_frame(wbuf, transform_wide)
      
      # 前処理済み画像をモデル入力バッファに設定
      self.model.setInputBuffer("mainCamera", main_camera_input)
      self.model.setInputBuffer("zoomCamera", zoom_camera_input)
      
      cloudlog.debug(f"Camera inputs processed: main={main_camera_input.shape}, zoom={zoom_camera_input.shape}")
      
    except Exception as e:
      cloudlog.error(f"Error processing camera inputs: {e}")
      # エラー時はゼロで埋めたダミー画像を使用（動的サイズ対応）
      dummy_image = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
      self.model.setInputBuffer("mainCamera", dummy_image)
      self.model.setInputBuffer("zoomCamera", dummy_image)
    
    # ===== Step 2: 車両状態の前処理 =====
    try:
      # モデル設定から車両状態ベクターの次元数を取得
      car_state_dim = self.model_config['car_state_dim']
      # 車両状態データを前処理（正規化、次元調整）
      car_state_input = process_car_state(inputs.get('carState'), target_dim=car_state_dim)
      # 前処理済みデータをモデル入力バッファに設定
      self.model.setInputBuffer("carState", car_state_input)
      cloudlog.debug(f"CarState input processed: shape={car_state_input.shape}, values={car_state_input[0][:min(4, car_state_dim)]}")
      
    except Exception as e:
      cloudlog.error(f"Error processing carState input: {e}")
      # エラー時はゼロベクターを使用（動的次元対応）
      car_state_dim = self.model_config['car_state_dim']
      car_state_input = np.zeros((1, car_state_dim), dtype=np.float32)
      self.model.setInputBuffer("carState", car_state_input)
    
    # ===== Step 3: ナビゲーション情報の前処理 =====
    try:
      # モデル設定からナビゲーションベクターの次元数を取得
      nav_vector_dim = self.model_config['nav_vector_dim']
      # ナビゲーション特徴量を前処理（次元調整、値域制限）
      nav_vector_input = process_nav_vector(inputs.get('nav_features'), target_dim=nav_vector_dim)
      # 前処理済みデータをモデル入力バッファに設定
      self.model.setInputBuffer("navVector", nav_vector_input)
      cloudlog.debug(f"NavVector input processed: shape={nav_vector_input.shape}, nonzero={np.count_nonzero(nav_vector_input)}")
      
    except Exception as e:
      cloudlog.error(f"Error processing navVector input: {e}")
      # エラー時はゼロベクターを使用（動的次元対応）
      nav_vector_dim = self.model_config['nav_vector_dim']
      nav_vector_input = np.zeros((1, nav_vector_dim), dtype=np.float32)
      self.model.setInputBuffer("navVector", nav_vector_input)

    # ===== Step 4: 準備のみの場合は早期リターン =====
    if prepare_only:
      # フレームドロップ等で推論をスキップする場合
      return None

    # ===== Step 5: モデル推論の実行 =====
    cloudlog.debug("E2E model executing with real inputs...")
    self.model.execute()  # GPU上でONNXモデルを実行
    
    # ===== Step 6: モデル出力の取得と解析 =====
    model_output = self.model.output  # カスタム出力バッファから結果を取得
    outputs = self.slice_outputs(model_output)  # 出力を辞書形式に変換
    return outputs
    

class DummyVisionBuf:
  """デモモード用のダミーVisionBufクラス"""
  def __init__(self, height, width, frame_id=0):
    self.height = height
    self.width = width
    self.frame_id = frame_id
    # YUV420フォーマットのダミーデータ生成
    yuv_height = height + height // 2  # Y + U/V planes
    self._yuv_data = np.random.randint(0, 255, (yuv_height, width), dtype=np.uint8)
    
    # VisionBufと同じdata属性を提供
    self.data = self._yuv_data.tobytes()
    
  def get_yuv_420(self):
    """YUV420データを返す"""
    return self._yuv_data


def generate_dummy_buffer(height, width, frame_id=0):
  """
  デモモード用のダミーカメラバッファを生成
  
  Args:
    height: フレーム高さ
    width: フレーム幅
    frame_id: フレームID
    
  Returns:
    DummyVisionBuf: ダミーフレームデータ
  """
  return DummyVisionBuf(height, width, frame_id)


class DummyFrameMeta:
  """デモモード用のダミーFrameMetaクラス"""
  def __init__(self, frame_id=0, timestamp_sof=0):
    self.frame_id = frame_id
    self.timestamp_sof = timestamp_sof


def main(demo=False):
  """
  E2Eモデルデーモンのメイン関数
  
  処理フロー:
  1. プロセス設定とログ初期化
  2. OpenCLコンテキストとE2Eモデルの初期化
  3. カメラクライアントの設定と接続
  4. メッセージング（PubMaster/SubMaster）の初期化
  5. メインループでの推論実行とメッセージ送信
  
  Args:
    demo: bool - デモモード（CarParamsを自動生成）
      - True: シミュレータモード（get_demo_car_params使用）
      - False: 実機モード（実際のCarParams読み込み）
  """
  cloudlog.warning("e2emodeld init")

  # ===== 定数定義（カメラ解像度） =====
  H, W = 874, 1164  # OpenPilot標準カメラ解像度

  # ===== プロセス設定の初期化 =====
  sentry.set_tag("daemon", PROCESS_NAME)  # Sentryエラー追跡用タグ設定
  cloudlog.bind(daemon=PROCESS_NAME)      # ログにプロセス名をバインド
  setproctitle(PROCESS_NAME)              # プロセス名を設定（psコマンドで確認可能）
  config_realtime_process(7, 54)          # リアルタイムプロセス設定（優先度7、CPU54番）

  # ===== 実行環境の判定と通知 =====
  if demo:
    cloudlog.warning("🎮 E2E Demo Mode: Using simulated car parameters")
  else:
    cloudlog.warning("🚗 E2E Real Mode: Using actual car parameters")

  # ===== OpenCLコンテキストとE2Eモデルの初期化 =====
  try:
    cloudlog.warning("setting up CL context")
    cl_context = CLContext()  # OpenCL実行コンテキスト（GPU処理用）
    cloudlog.warning("CL context ready; loading E2E model")
    model = E2EModelState(cl_context)  # E2Eモデルの初期化
    cloudlog.warning("E2E model loaded, e2emodeld starting")
  except Exception as e:
    cloudlog.error(f"Failed to initialize E2E model: {e}")
    import traceback
    cloudlog.error(f"E2E model initialization error traceback: {traceback.format_exc()}")
    raise

  # ===== カメラクライアントの設定（シミュレータ・実機対応） =====
  try:
    cloudlog.warning("Setting up vision clients...")
    
    # カメラストリームの自動検出（環境に応じて適応）
    timeout_count = 0
    max_timeout = 50  # 5秒間の試行
    
    while True:
      available_streams = VisionIpcClient.available_streams("camerad", block=False)
      if available_streams:
        # ストリーム設定の判定
        use_extra_client = (VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and 
                           VisionStreamType.VISION_STREAM_ROAD in available_streams)
        main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
        
        # 検出されたストリーム情報をログ出力（整数値対応）
        try:
          stream_names = [stream.name if hasattr(stream, 'name') else str(stream) for stream in available_streams]
        except:
          stream_names = [str(stream) for stream in available_streams]
        cloudlog.warning(f"📹 Detected camera streams: {stream_names}")
        break
        
      timeout_count += 1
      if timeout_count >= max_timeout:
        cloudlog.error("⚠️ Timeout waiting for camera streams")
        if demo:
          cloudlog.warning("🎮 Demo mode: Proceeding without camera streams (may use dummy data)")
          # デモモードでは続行を許可
          available_streams = []
          use_extra_client = False
          main_wide_camera = True
          break
        else:
          raise RuntimeError("Camera streams not available in real mode")
      
      time.sleep(0.1)

    # カメラクライアントの初期化
    if available_streams:
      vipc_client_main_stream = (VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera 
                                else VisionStreamType.VISION_STREAM_ROAD)
      vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, cl_context)
      vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, cl_context)
      
      cloudlog.warning(f"📷 Vision config: main_wide_camera={main_wide_camera}, use_extra_client={use_extra_client}")

      # メインカメラ接続
      connect_timeout = 0
      while not vipc_client_main.connect(False):
        connect_timeout += 1
        if connect_timeout > 100:  # 10秒タイムアウト
          if demo:
            cloudlog.warning("🎮 Demo mode: Main camera connection timeout, using dummy frames")
            break
          else:
            raise RuntimeError("Main camera connection failed")
        time.sleep(0.1)
      
      # 追加カメラ接続（利用可能な場合）
      if use_extra_client:
        extra_timeout = 0
        while not vipc_client_extra.connect(False):
          extra_timeout += 1
          if extra_timeout > 100:  # 10秒タイムアウト
            cloudlog.warning("Extra camera connection timeout, proceeding with main camera only")
            use_extra_client = False
            break
          time.sleep(0.1)

      # 接続成功の確認
      if vipc_client_main.connect(False):
        cloudlog.warning(f"✅ Main camera connected: {vipc_client_main.buffer_len} buffers "
                        f"({vipc_client_main.width} x {vipc_client_main.height})")
      if use_extra_client and vipc_client_extra.connect(False):
        cloudlog.warning(f"✅ Extra camera connected: {vipc_client_extra.buffer_len} buffers "
                        f"({vipc_client_extra.width} x {vipc_client_extra.height})")
    else:
      # カメラなしモード（デモ用）
      cloudlog.warning("🎮 No camera mode: Using dummy camera clients")
      vipc_client_main = None
      vipc_client_extra = None
      use_extra_client = False
      main_wide_camera = True
      
  except Exception as e:
    cloudlog.error(f"Failed to setup vision clients: {e}")
    if demo:
      cloudlog.warning("🎮 Demo mode: Continuing despite vision setup failure")
      vipc_client_main = None
      vipc_client_extra = None
      use_extra_client = False
      main_wide_camera = True
    else:
      raise

  # messaging - E2Eモデル専用のメッセージトピック（e2eOutputのみ）
  try:
    pm = PubMaster(["e2eOutput"])
    sm = SubMaster(["carState", "navInstruction"])
  except Exception as e:
    cloudlog.error(f"Failed to setup messaging: {e}")
    raise
  
  # デバッグ: 初期化完了をログ出力
  cloudlog.warning(f"E2E modeld initialized: frequency={E2E_MODEL_FREQ}Hz, messaging ready")

  params = Params()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  nav_features = np.zeros(ModelConstants.NAV_FEATURE_LEN, dtype=np.float32)
  nav_instructions = np.zeros(ModelConstants.NAV_INSTRUCTION_LEN, dtype=np.float32)
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()

  if demo:
    CP = get_demo_car_params()
  else:
    with car.CarParams.from_bytes(params.get("CarParams", block=True)) as msg:
      CP = msg
  cloudlog.info("e2emodeld got CarParams: %s", CP.carName)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  steer_delay = CP.steerActuatorDelay + .2

  DH = DesireHelper()

  cloudlog.warning("E2E model main loop starting")
  
  # E2E専用の更新頻度制御とフレーム処理
  last_e2e_update_time = 0.0
  e2e_update_interval = 1.0 / E2E_MODEL_FREQ  # 10Hz間隔

  while True:
    current_time = time.monotonic()
    
    # E2E更新頻度制御（負荷分散のため）
    if current_time - last_e2e_update_time < e2e_update_interval:
      time.sleep(0.001)  # 短時間スリープ
      continue
    
    # ===== カメラフレーム取得（シミュレータ・実機対応） =====
    try:
      # メインカメラフレーム処理
      if vipc_client_main is not None:
        # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
        while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          buf_main = vipc_client_main.recv()
          meta_main = FrameMeta(vipc_client_main)
          if buf_main is None:
            break

        if buf_main is None:
          if demo:
            cloudlog.debug("🎮 Demo mode: No main frame, using dummy data")
            # ダミーフレームバッファとメタデータ作成
            buf_main = generate_dummy_buffer(H, W, frame_id)
            meta_main = DummyFrameMeta(frame_id, int(current_time * 1e9))
          else:
            cloudlog.error("vipc_client_main no frame")
            continue
      else:
        # カメラなしモード（デモ専用）
        if demo:
          buf_main = generate_dummy_buffer(H, W, frame_id)
          meta_main = DummyFrameMeta(frame_id, int(current_time * 1e9))
        else:
          cloudlog.error("No main camera available in real mode")
          continue

      # 追加カメラフレーム処理
      if use_extra_client and vipc_client_extra is not None:
        # Keep receiving extra frames until frame id matches main camera
        while True:
          buf_extra = vipc_client_extra.recv()
          meta_extra = FrameMeta(vipc_client_extra)
          if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
            break

        if buf_extra is None:
          if demo:
            cloudlog.debug("🎮 Demo mode: No extra frame, using main frame")
            buf_extra = buf_main
            meta_extra = meta_main
          else:
            cloudlog.error("vipc_client_extra no frame")
            continue

        # フレーム同期チェック
        if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
          cloudlog.warning("frames out of sync! main: {} ({:.5f}), extra: {} ({:.5f})".format(
            meta_main.frame_id, meta_main.timestamp_sof / 1e9,
            meta_extra.frame_id, meta_extra.timestamp_sof / 1e9))
          if not demo:
            continue  # 実機では同期エラー時はスキップ
      else:
        # シングルカメラモード
        buf_extra = buf_main
        meta_extra = meta_main

    except Exception as e:
      cloudlog.error(f"Camera frame processing error: {e}")
      if demo:
        cloudlog.warning("🎮 Demo mode: Continuing with dummy frames after error")
        # エラー時のフォールバック
        buf_main = generate_dummy_buffer(H, W, frame_id)
        buf_extra = buf_main
        meta_main = DummyFrameMeta(frame_id, int(current_time * 1e9))
        meta_extra = meta_main
      else:
        continue

    sm.update(0)
    desire = DH.desire
    # デフォルト値を使用（元のSubMasterから削除されたデータ）
    is_rhd = True  # 右ハンドル車として設定
    frame_id = meta_main.frame_id  # メインカメラのフレームIDを使用
    lateral_control_params = np.array([sm["carState"].vEgo, steer_delay], dtype=np.float32)
    
    # キャリブレーションデータがないため、単位行列を使用
    if not live_calib_seen:
      model_transform_main = np.eye(3, dtype=np.float32)
      model_transform_extra = np.eye(3, dtype=np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    # Enable/disable nav features - navInstructionのみ使用
    timestamp_llk = 0  # navModelを使用していないため0に設定
    nav_valid = sm.valid["navInstruction"]  # navInstructionの有効性をチェック
    nav_enabled = nav_valid  # ExperimentalModeチェックを簡略化

    if not nav_enabled:
      nav_features[:] = 0
      nav_instructions[:] = 0

    # navModelを使用しないため、nav_featuresは常に0
    nav_features[:] = 0

    if nav_enabled and sm.updated["navInstruction"]:
      nav_instructions[:] = 0
      for maneuver in sm["navInstruction"].allManeuvers:
        distance_idx = 25 + int(maneuver.distance / 20)
        direction_idx = 0
        if maneuver.modifier in ("left", "slight left", "sharp left"):
          direction_idx = 1
        if maneuver.modifier in ("right", "slight right", "sharp right"):
          direction_idx = 2
        if 0 <= distance_idx < 50:
          nav_instructions[distance_idx*3 + direction_idx] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0
    if prepare_only:
      cloudlog.error(f"skipping E2E model eval. Dropped {vipc_dropped_frames} frames")

    inputs: Dict[str, np.ndarray] = {
      'desire': vec_desire,
      'traffic_convention': traffic_convention,
      'lateral_control_params': lateral_control_params,
      'nav_features': nav_features,
      'nav_instructions': nav_instructions}

    # デバッグ: 入力データの基本統計を出力（推論が実行される場合のみ）
    if not prepare_only:
      cloudlog.debug(f"E2E inputs - desire: {vec_desire[:3]}, nav_feat_mean: {np.mean(nav_features):.6f}, nav_inst_sum: {np.sum(nav_instructions)}")

    mt1 = time.perf_counter()
    model_output = model.run(buf_main, buf_extra, model_transform_main, model_transform_extra, inputs, prepare_only)
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      cloudlog.debug(f"E2E model execution time: {model_execution_time:.4f}s")
      
      try:
        # E2Eモデル（checkpoint_epoch_19_best.onnx）の実際の出力を取得
        # モデル出力: control_output [batch_size, 2] - [aEgo, steeringTorque]
        e2e_aEgo = 0.0
        e2e_steeringTorque = 0.0
        
        # カスタムモデルの 'control_output' キーを確認
        if 'control_output' in model_output:
          control_outputs = model_output['control_output']
          cloudlog.debug(f"E2E ONNX control_output shape: {control_outputs.shape}")
          
          # control_outputから値を取得 [1, 2] -> [aEgo, steeringTorque]
          if hasattr(control_outputs, 'flatten') and len(control_outputs.flatten()) >= 2:
            flat_outputs = control_outputs.flatten()
            e2e_aEgo = float(flat_outputs[0])
            e2e_steeringTorque = float(flat_outputs[1])
            cloudlog.debug(f"E2E parsed from control_output: aEgo={e2e_aEgo:.6f}, steeringTorque={e2e_steeringTorque:.6f}")
          else:
            cloudlog.warning(f"E2E model control_output format unexpected: {control_outputs.shape}")
            
        # ONNXモデルの出力を処理（デバッグ用）
        elif 'raw_pred' in model_output:
          raw_prediction = model_output['raw_pred']
          cloudlog.debug(f"E2E ONNX raw prediction shape: {raw_prediction.shape}")
          
          # ONNXモデルの出力は [batch_size, 2] = [aEgo, steeringTorque]
          if len(raw_prediction) >= 2:
            e2e_aEgo = float(raw_prediction[0])           # 第1要素: 加速度指令
            e2e_steeringTorque = float(raw_prediction[1]) # 第2要素: ステアリングトルク指令
          else:
            cloudlog.warning(f"E2E model output insufficient: {len(raw_prediction)} < 2")
            
        # パースされた出力がある場合（フォールバック）
        elif 'outputs' in model_output:
          outputs = model_output['outputs']
          cloudlog.debug(f"E2E ONNX parsed outputs shape: {outputs.shape}")
          
          # control_outputから値を取得
          if hasattr(outputs, 'flatten') and len(outputs.flatten()) >= 2:
            flat_outputs = outputs.flatten()
            e2e_aEgo = float(flat_outputs[0])
            e2e_steeringTorque = float(flat_outputs[1])
          else:
            cloudlog.warning("E2E model parsed output format unexpected")
            
        else:
          cloudlog.warning(f"E2E model output format not recognized. Available keys: {list(model_output.keys())}")
        
        # E2E専用の出力データを作成
        # ONNXモデル checkpoint_epoch_19_best.onnx の control_output [2] に対応
        e2e_output_data = {
          'aEgo': e2e_aEgo,                    # 加速度指令 (m/s²)
          'steeringTorque': e2e_steeringTorque, # ステアリングトルク指令 (Nm)
        }
        
        cloudlog.debug(f"E2E ONNX model predictions - aEgo: {e2e_aEgo:.6f} m/s², steeringTorque: {e2e_steeringTorque:.6f} Nm, execTime: {model_execution_time:.3f}ms")
        
        # E2E専用の出力をログに記録
        cloudlog.warning(f"E2E OUTPUT: aEgo={e2e_aEgo:.4f}, steeringTorque={e2e_steeringTorque:.4f}")
        
        # rlogに記録するためのメッセージ送信（e2eOutputのみ）
        if pm is not None:
          import cereal.messaging as messaging
          
          # e2eOutputメッセージ（ミニマル構造）
          e2e_out_msg = messaging.new_message('e2eOutput')
          e2e_out_msg.e2eOutput.aEgo = e2e_aEgo
          e2e_out_msg.e2eOutput.steeringTorque = e2e_steeringTorque
          e2e_out_msg.e2eOutput.timestamp = int(time.time_ns())
          e2e_out_msg.e2eOutput.isValid = True
          
          # 重要: メッセージレベルのvalidフラグも設定
          e2e_out_msg.valid = True
          
          pm.send('e2eOutput', e2e_out_msg)
          
          # デバッグログ: メッセージ送信確認
          cloudlog.debug(f"E2E message sent: aEgo={e2e_aEgo:.4f}, steeringTorque={e2e_steeringTorque:.4f}, isValid=True")
        
        # E2E更新時間を記録（負荷分散制御用）
        last_e2e_update_time = current_time
        
      except Exception as e:
        cloudlog.error(f"Error processing E2E model output: {e}")
        import traceback
        cloudlog.error(f"E2E error traceback: {traceback.format_exc()}")
        
        # エラー時でも無効なメッセージを送信（デバッグ用）
        try:
          e2e_out_msg = messaging.new_message('e2eOutput')
          e2e_out_msg.e2eOutput.aEgo = 0.0
          e2e_out_msg.e2eOutput.steeringTorque = 0.0
          e2e_out_msg.e2eOutput.timestamp = int(time.time_ns())
          e2e_out_msg.e2eOutput.isValid = False
          
          # エラー時: メッセージレベルのvalidも無効に設定
          e2e_out_msg.valid = False
          
          pm.send('e2eOutput', e2e_out_msg)
          cloudlog.debug("E2E error: sent invalid message")
        except Exception as msg_error:
          cloudlog.error(f"Failed to send error message: {msg_error}")

    # フレームIDの更新（次回のフレームドロップ検出用）
    last_vipc_frame_id = meta_main.frame_id


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
    parser = argparse.ArgumentParser(description='E2E自動運転モデル実行デーモン')
    parser.add_argument('--demo', action='store_true', 
                       help='デモモード（CarParams自動生成、実車接続不要）')
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
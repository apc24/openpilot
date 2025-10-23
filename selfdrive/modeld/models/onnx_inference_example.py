# -*- coding: utf-8 -*-
"""
ONNX変換後のモデル使用例

このスクリプトは変換されたONNXモデルを使用して
推論を実行する方法を示します。
"""

import numpy as np
import time
from pathlib import Path
import sys
from typing import Dict

try:
    import onnxruntime as ort
except ImportError:
    print("エラー: onnxruntimeがインストールされていません")
    print("以下のコマンドでインストールしてください:")
    print("pip install onnxruntime")
    sys.exit(1)


class ONNXAutonomousDrivingModel:
    """
    ONNX形式の自動運転モデルのラッパークラス
    
    変換されたONNXモデルを簡単に使用するためのクラスです。
    """
    
    def __init__(self, model_path: str, providers: list = None):
        """
        ONNXモデルを初期化
        
        Args:
            model_path (str): ONNXモデルファイルのパス
            providers (list): 使用するプロバイダー（実行環境）のリスト
        """
        self.model_path = model_path
        
        # デフォルトプロバイダーの設定
        if providers is None:
            providers = ['CPUExecutionProvider']
            # GPU利用可能であれば追加
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
        
        print(f"ONNXモデルを読み込み中: {model_path}")
        print(f"使用プロバイダー: {providers}")
        
        try:
            # ONNXランタイムセッションを作成
            self.session = ort.InferenceSession(
                model_path, 
                providers=providers
            )
            
            # 入力・出力情報を取得
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"✓ モデルの読み込み完了")
            print(f"入力: {self.input_names}")
            print(f"出力: {self.output_names}")
            
        except Exception as e:
            print(f"エラー: ONNXモデルの読み込みに失敗しました: {str(e)}")
            raise
    
    def predict(self, input_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        推論を実行
        
        Args:
            input_data (Dict[str, np.ndarray]): 入力データの辞書
            
        Returns:
            np.ndarray: 予測結果（制御信号）
        """
        try:
            # 推論実行
            outputs = self.session.run(self.output_names, input_data)
            return outputs[0]  # 制御出力を返す
            
        except Exception as e:
            print(f"エラー: 推論実行中に問題が発生しました: {str(e)}")
            raise
    
    def get_input_shapes(self) -> Dict[str, tuple]:
        """入力の形状情報を取得"""
        shapes = {}
        for input_info in self.session.get_inputs():
            shapes[input_info.name] = input_info.shape
        return shapes
    
    def get_output_shapes(self) -> Dict[str, tuple]:
        """出力の形状情報を取得"""
        shapes = {}
        for output_info in self.session.get_outputs():
            shapes[output_info.name] = output_info.shape
        return shapes


def create_sample_input_data(batch_size: int = 1) -> Dict[str, np.ndarray]:
    """
    サンプル入力データを生成
    
    Args:
        batch_size (int): バッチサイズ
        
    Returns:
        Dict[str, np.ndarray]: サンプル入力データ
    """
    # 実際の使用時は、これらをセンサーからの実データに置き換える
    sample_data = {
        # 車両状態（速度、加速度、ヨー角など8次元）
        'carState': np.random.randn(batch_size, 8).astype(np.float32),
        # メインカメラ画像（3チャンネル、224x224）
        'mainCamera': np.random.randn(batch_size, 3, 224, 224).astype(np.float32),

        # ズームカメラ画像（3チャンネル、224x224）
        'zoomCamera': np.random.randn(batch_size, 3, 224, 224).astype(np.float32),

        # ナビゲーション情報（150次元ベクトル）
        'navVector': np.random.randn(batch_size, 150).astype(np.float32)
    }
    
    return sample_data


def benchmark_performance(model: ONNXAutonomousDrivingModel, num_runs: int = 100):
    """
    モデルのパフォーマンスをベンチマーク
    
    Args:
        model (ONNXAutonomousDrivingModel): テスト対象のモデル
        num_runs (int): 実行回数
    """
    print(f"\n=== パフォーマンステスト ({num_runs}回実行) ===")
    
    # ウォームアップ（最初の数回は除外）
    warmup_data = create_sample_input_data(1)
    for _ in range(5):
        model.predict(warmup_data)
    
    # 実際のベンチマーク
    times = []
    test_data = create_sample_input_data(1)
    
    for i in range(num_runs):
        start_time = time.time()
        result = model.predict(test_data)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # ミリ秒
        times.append(inference_time)
        
        if i % 20 == 0:
            print(f"実行 {i+1}/{num_runs}: {inference_time:.2f}ms")
    
    # 統計情報の表示
    times = np.array(times)
    print(f"\n📊 パフォーマンス統計:")
    print(f"平均推論時間: {np.mean(times):.2f}ms")
    print(f"最小推論時間: {np.min(times):.2f}ms")
    print(f"最大推論時間: {np.max(times):.2f}ms")
    print(f"標準偏差: {np.std(times):.2f}ms")
    print(f"FPS (理論値): {1000/np.mean(times):.1f}")


def compare_batch_sizes(model: ONNXAutonomousDrivingModel):
    """
    異なるバッチサイズでのパフォーマンス比較
    
    Args:
        model (ONNXAutonomousDrivingModel): テスト対象のモデル
    """
    print(f"\n=== バッチサイズ別パフォーマンス比較 ===")
    
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        print(f"\nバッチサイズ {batch_size}:")
        
        # テストデータ準備
        test_data = create_sample_input_data(batch_size)
        
        # 複数回実行して平均時間を測定
        times = []
        for _ in range(20):
            start_time = time.time()
            result = model.predict(test_data)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        per_sample_time = avg_time / batch_size
        
        print(f"  バッチ全体: {avg_time:.2f}ms")
        print(f"  サンプル当たり: {per_sample_time:.2f}ms")
        print(f"  スループット: {1000/per_sample_time:.1f} samples/sec")


def main():
    """メイン処理"""
    print("=== ONNX自動運転モデル 使用例 ===\n")
    
    # モデルファイルのパスを設定
    # この部分を実際のONNXファイルパスに変更してください
    model_path = "checkpoint_epoch_5_best.onnx"
    
    # モデルファイルの存在確認
    if not Path(model_path).exists():
        print(f"エラー: ONNXモデルファイルが見つかりません: {model_path}")
        print("\n利用可能なONNXファイル:")
        
        models_dir = Path("models")
        if models_dir.exists():
            onnx_files = list(models_dir.glob("*.onnx"))
            if onnx_files:
                for onnx_file in onnx_files:
                    print(f"  - {onnx_file}")
                print(f"\n上記のいずれかのファイルパスに変更してください。")
            else:
                print("  見つかりませんでした")
                print("\n最初にPyTorchモデルをONNX形式に変換してください:")
                print("python src/tools/convert_to_onnx.py models/checkpoint_epoch_5_best.pt")
        
        return
    
    try:
        # ONNXモデルを初期化
        model = ONNXAutonomousDrivingModel(model_path)
        
        # モデル情報の表示
        print(f"\n📋 モデル情報:")
        print(f"入力形状: {model.get_input_shapes()}")
        print(f"出力形状: {model.get_output_shapes()}")
        
        # 単発推論のテスト
        print(f"\n🔮 単発推論テスト:")
        sample_input = create_sample_input_data(1)
        
        start_time = time.time()
        result = model.predict(sample_input)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"入力データ形状:")
        for name, data in sample_input.items():
            print(f"  {name}: {data.shape}")
        
        print(f"\n出力結果:")
        print(f"  制御信号: {result.shape}")
        print(f"  値: {result[0]}")  # 最初のサンプルの制御値
        print(f"  推論時間: {inference_time:.2f}ms")
        
        # パフォーマンステスト
        benchmark_performance(model)
        
        # バッチサイズ比較
        compare_batch_sizes(model)
        
        print(f"\n🎉 すべてのテストが正常に完了しました！")
        
    except Exception as e:
        print(f"エラー: {str(e)}")
        return


if __name__ == '__main__':
    main()
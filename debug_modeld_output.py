#!/usr/bin/env python3
"""
標準OpenPilot ModelV2出力監視ツール

このスクリプトは標準OpenPilotのmodeld（SuperComboモデル）が出力する
modelV2メッセージを監視し、リアルタイムでログ出力します。

用途:
- 標準OpenPilotモデルの動作確認
- E2Eモデルとの性能比較・検証
- システム全体の健全性チェック
- フレームドロップやモデル実行時間の監視

対象メッセージ: modelV2（標準OpenPilotビジョンモデル）
更新頻度: 20Hz（標準OpenPilot設定）

注意: これはE2Eモデル（e2eOutput）とは別のメッセージです。
E2Eモデルの出力を監視する場合は debug_e2e_output.py を使用してください。

実行方法:
python debug_modeld_output.py

Ctrl+Cで停止
"""
import time
from cereal.messaging import SubMaster
from openpilot.common.swaglog import cloudlog

def main():
    """
    標準OpenPilotモデル（SuperCombo）の出力を監視するメイン関数
    
    監視内容:
    - 位置予測 (position.x, position.y)
    - 速度予測 (velocity.x, velocity.y)  
    - 加速度予測 (acceleration.x, acceleration.y)
    - モデル実行時間 (modelExecutionTime)
    - フレームID (frameId)
    
    この情報は車線維持、車線変更、前方衝突警告などに使用されます。
    """
    print("=" * 60)
    print("🚗 Standard OpenPilot ModelV2 Output Monitor")
    print("=" * 60)
    print("📊 監視対象: modelV2メッセージ（標準SuperComboモデル）")
    print("🔄 更新頻度: 20Hz")
    print("🎯 用途: 標準モデル動作確認・E2E比較・システム健全性チェック")
    print("⚠️  注意: E2Eモデル出力ではありません（E2E監視 → debug_e2e_output.py）")
    print("=" * 60)
    print()
    
    # modelV2メッセージを監視（標準OpenPilotビジョンモデル）
    sm = SubMaster(['modelV2'])
    
    message_count = 0  # 受信メッセージ数カウンター
    last_frame_id = -1  # フレームドロップ検出用
    start_time = time.time()  # 監視開始時刻
    
    while True:
        # 100msタイムアウトでメッセージ更新をチェック
        sm.update(timeout=100)  # 100ms timeout
        
        # modelV2メッセージが更新された場合の処理
        if sm.updated['modelV2']:
            model_data = sm['modelV2']
            message_count += 1
            current_time = time.time()
            
            # ===== 基本メタデータの取得 =====
            frame_id = model_data.frameId                        # フレーム通番
            exec_time = model_data.modelExecutionTime * 1000     # 実行時間をmsに変換
            
            # フレームドロップの検出
            if last_frame_id != -1 and frame_id != last_frame_id + 1:
                dropped_frames = frame_id - last_frame_id - 1
                print(f"⚠️  FRAME DROP DETECTED: {dropped_frames} frames dropped (last: {last_frame_id}, current: {frame_id})")
            last_frame_id = frame_id
            
            # ===== モデル予測データの解析と表示 =====
            # 優先順位: 位置 → 速度 → 加速度 → 基本情報のみ
            
            # 1. 位置データ（車両の将来位置予測）
            if hasattr(model_data, 'position') and len(model_data.position.x) > 0:
                pos_x = model_data.position.x[0]  # 前方方向の位置 [m]
                pos_y = model_data.position.y[0]  # 横方向の位置 [m]
                print(f"📍 POSITION: x={pos_x:+7.4f}m, y={pos_y:+7.4f}m | "
                      f"⏱️  exec={exec_time:6.2f}ms | 🎞️  frame={frame_id:06d} | "
                      f"📨 count={message_count:04d}")
            
            # 2. 速度データ（車両の将来速度予測）
            elif hasattr(model_data, 'velocity') and len(model_data.velocity.x) > 0:
                vel_x = model_data.velocity.x[0]  # 前方方向の速度 [m/s]
                vel_y = model_data.velocity.y[0]  # 横方向の速度 [m/s]
                print(f"🏃 VELOCITY: x={vel_x:+7.4f}m/s, y={vel_y:+7.4f}m/s | "
                      f"⏱️  exec={exec_time:6.2f}ms | 🎞️  frame={frame_id:06d} | "
                      f"📨 count={message_count:04d}")
            
            # 3. 加速度データ（車両の将来加速度予測）
            elif hasattr(model_data, 'acceleration') and len(model_data.acceleration.x) > 0:
                accel_x = model_data.acceleration.x[0]                                    # 前方加速度 [m/s²]
                accel_y = model_data.acceleration.y[0] if len(model_data.acceleration.y) > 0 else 0.0  # 横方向加速度 [m/s²]
                print(f"🚀 ACCELERATION: x={accel_x:+7.4f}m/s², y={accel_y:+7.4f}m/s² | "
                      f"⏱️  exec={exec_time:6.2f}ms | 🎞️  frame={frame_id:06d} | "
                      f"📨 count={message_count:04d}")
            
            # 4. フォールバック（基本メタデータのみ）
            else:
                elapsed_time = current_time - start_time
                print(f"📊 BASIC INFO: exec={exec_time:6.2f}ms | 🎞️  frame={frame_id:06d} | "
                      f"📨 count={message_count:04d} | ⏰ elapsed={elapsed_time:6.1f}s")
            
            # パフォーマンス警告の表示
            if exec_time > 50:  # 50ms以上の場合は警告
                print(f"⚠️  HIGH EXECUTION TIME: {exec_time:.2f}ms (> 50ms threshold)")
        
        # 20Hz相当の更新間隔でスリープ
        time.sleep(0.05)  # 20Hz更新（標準OpenPilotモデルの周波数に合わせる）

if __name__ == "__main__":
    """
    標準OpenPilotモデル監視ツールのエントリーポイント
    
    このスクリプトは標準OpenPilotのSuperComboモデルの出力を監視します。
    E2Eモデルの監視には debug_e2e_output.py を使用してください。
    
    使用場面:
    - 標準OpenPilotの動作確認
    - E2Eモデルとの比較分析
    - パフォーマンステスト
    - システム健全性チェック
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("🛑 Standard OpenPilot ModelV2 monitor stopped by user")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("💡 Troubleshooting:")
        print("   - OpenPilotが実行中か確認してください")
        print("   - modeldプロセスが動作しているか確認してください")
        print("   - modelV2メッセージが送信されているか確認してください")
        raise
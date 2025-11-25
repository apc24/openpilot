#!/usr/bin/env python3

"""
E2E（End-to-End）モデル出力監視ツール

このスクリプトはカスタムE2Eモデル（checkpoint_epoch_5_best.onnx）が出力する
e2eOutputメッセージを監視し、リアルタイムで制御信号をログ出力します。

用途:
- E2Eモデルの制御信号確認（加速度・ステアリング）
- モデル推論の動作状況監視
- 固定値テストモードの検証
- E2E制御の有効性チェック

対象メッセージ: e2eOutput（カスタムE2Eモデル専用）
更新頻度: 10Hz（E2E_MODEL_FREQ設定）

注意: これは標準OpenPilotモデル（modelV2）とは別のメッセージです。
標準モデルの出力を監視する場合は debug_modeld_output.py を使用してください。

前提条件:
- e2emodeld.pyが実行中であること
- カスタムONNXモデルが正常に読み込まれていること
- カメラデーモン（camerad）が動作していること

実行方法:
python debug_e2e_output.py

Ctrl+Cで停止
"""

import time
import cereal.messaging as messaging

def main():
    """
    E2Eモデル出力を監視するメイン関数
    
    監視内容:
    - aEgo: 加速度制御信号 [m/s²] - 正値=加速、負値=減速
    - steeringTorque: ステアリングトルク制御信号 [Nm] - 正値=右、負値=左
    - timestamp: メッセージタイムスタンプ [nanosecond]
    - isValid: メッセージ有効性フラグ [boolean]
    
    E2E制御信号は車両の実際の制御に使用される可能性があります。
    固定値テストモード（E2E_FIXED_TEST=1）では固定値が出力されます。
    """
    print("=" * 70)
    print("🧠 E2E (End-to-End) Model Output Monitor")
    print("=" * 70)
    print("📊 監視対象: e2eOutputメッセージ（カスタムE2Eモデル）")
    print("🔄 更新頻度: 10Hz")
    print("🎯 用途: E2E制御信号監視・固定値テスト検証・モデル動作確認")
    print("⚠️  注意: 標準OpenPilotモデルではありません（標準監視 → debug_modeld_output.py）")
    print("📋 出力形式: [時刻] 加速度指令, ステアリングトルク, タイムスタンプ, 有効性")
    print("=" * 70)
    print()
    
    # e2eOutputメッセージのみを購読（カスタムE2Eモデル専用）
    sm = messaging.SubMaster(['e2eOutput'])
    
    # 監視状態の初期化
    last_e2e_output_time = 0      # 最後のメッセージ受信時刻
    message_count = 0             # 受信メッセージ数カウンター
    start_time = time.time()      # 監視開始時刻
    valid_count = 0               # 有効メッセージ数
    invalid_count = 0             # 無効メッセージ数
    
    try:
        while True:
            # 1秒タイムアウトでメッセージ更新をチェック（E2Eは10Hzなので余裕を持たせる）
            sm.update(timeout=1000)  # 1秒タイムアウト
            current_time = time.time()
            
            # ===== e2eOutputメッセージの処理 =====
            if sm.updated['e2eOutput']:
                e2e_data = sm['e2eOutput']
                message_count += 1
                
                # E2E制御信号の取得
                aEgo = e2e_data.aEgo                    # 加速度指令 [m/s²]
                steeringTorque = e2e_data.steeringTorque # ステアリングトルク指令 [Nm]
                timestamp = e2e_data.timestamp           # タイムスタンプ [nanosecond]
                isValid = e2e_data.isValid              # 有効性フラグ
                
                # 有効性統計の更新
                if isValid:
                    valid_count += 1
                    status_icon = "✅"
                else:
                    invalid_count += 1
                    status_icon = "❌"
                
                # E2E制御信号の詳細ログ出力
                print(f"[{time.strftime('%H:%M:%S')}] {status_icon} E2E: "
                      f"accel={aEgo:+7.4f}m/s² | "
                      f"steer={steeringTorque:+7.4f}Nm | "
                      f"timestamp={timestamp} | "
                      f"valid={isValid} | "
                      f"count={message_count:04d}")
                
                # 異常値の警告表示
                if abs(aEgo) > 5.0:  # 加速度が±5m/s²を超える場合
                    print(f"⚠️  HIGH ACCELERATION: {aEgo:.4f} m/s² (>±5.0 threshold)")
                
                if abs(steeringTorque) > 10.0:  # ステアリングトルクが±10Nmを超える場合
                    print(f"⚠️  HIGH STEERING TORQUE: {steeringTorque:.4f} Nm (>±10.0 threshold)")
                
                # 固定値テストモードの検出
                # 同じ値が連続で出力される場合は固定値モードの可能性
                if hasattr(main, 'prev_aEgo') and hasattr(main, 'prev_steeringTorque'):
                    if (aEgo == main.prev_aEgo and steeringTorque == main.prev_steeringTorque and 
                        aEgo != 0.0 and steeringTorque != 0.0):
                        print(f"🔧 FIXED VALUE MODE DETECTED: Same values repeated")
                
                main.prev_aEgo = aEgo
                main.prev_steeringTorque = steeringTorque
                    
                last_e2e_output_time = current_time
                
            # ===== タイムアウト・無応答の監視 =====
            # 5秒間更新がない場合の警告（E2Eモデルが停止している可能性）
            if current_time - last_e2e_output_time > 5:
                elapsed_no_update = current_time - last_e2e_output_time
                print(f"[{time.strftime('%H:%M:%S')}] ⚠️  WARNING: No E2E updates for {elapsed_no_update:.1f} seconds")
                print(f"💡 Possible causes:")
                print(f"   - e2emodeld.py not running")
                print(f"   - ONNX model loading failed")
                print(f"   - Camera daemon (camerad) stopped")
                print(f"   - E2E model execution errors")
                last_e2e_output_time = current_time  # 警告スパム防止
            
            # 10秒ごとに統計情報を表示
            if message_count > 0 and message_count % 100 == 0:  # 10Hz * 10秒 = 100メッセージ
                elapsed_time = current_time - start_time
                message_rate = message_count / elapsed_time
                valid_rate = (valid_count / message_count) * 100
                print(f"📊 STATS: {message_count} msgs in {elapsed_time:.1f}s | "
                      f"Rate: {message_rate:.1f}Hz | "
                      f"Valid: {valid_rate:.1f}% ({valid_count}/{message_count})")
                
    except KeyboardInterrupt:
        # 終了時の統計情報表示
        elapsed_time = time.time() - start_time
        if message_count > 0:
            message_rate = message_count / elapsed_time
            valid_rate = (valid_count / message_count) * 100
            print(f"\n" + "=" * 70)
            print(f"📊 Final Statistics:")
            print(f"   ⏰ Monitoring time: {elapsed_time:.1f} seconds")
            print(f"   📨 Total messages: {message_count}")
            print(f"   🔄 Average rate: {message_rate:.2f} Hz")
            print(f"   ✅ Valid messages: {valid_count} ({valid_rate:.1f}%)")
            print(f"   ❌ Invalid messages: {invalid_count}")
            print(f"=" * 70)
        print("🛑 E2E output monitor stopped by user")

if __name__ == "__main__":
    """
    E2Eモデル監視ツールのエントリーポイント
    
    このスクリプトはカスタムE2Eモデルの制御信号出力を監視します。
    標準OpenPilotモデルの監視には debug_modeld_output.py を使用してください。
    
    使用場面:
    - E2Eモデルの動作確認
    - 固定値テストモードの検証
    - 制御信号の異常値検出
    - E2E制御システムのデバッグ
    
    環境変数:
    - E2E_FIXED_TEST=1: 固定値テストモード
    """
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("💡 Troubleshooting:")
        print("   - e2emodeld.pyが実行中か確認してください")
        print("   - カスタムONNXモデルが正常に読み込まれているか確認してください")
        print("   - カメラデーモン（camerad）が動作しているか確認してください")
        print("   - e2eOutputメッセージが送信されているか確認してください")
        print("   - 固定値テストの場合: E2E_FIXED_TEST=1 を設定してください")
        raise
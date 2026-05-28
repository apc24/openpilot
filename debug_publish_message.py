# このスクリプトは custom.capnp の DebugPublishMessage を作成し、
# customReservedRawData0/1/2 サービスへ publish するデバッグ用ツールです。
#
# 実行前提:
# - openpilot および metadrive を ZMQ=1 で起動しておくこと
# - 送受信に必要なポートを開放しておくこと
#
# 基本例:
# ZMQ=1 poetry run python debug_publish_message.py --service customReservedRawData0 --text hello --interval-ms 200 --count 20
#
# 主な引数:
# --service      送信先サービス名（customReservedRawData0/1/2）
# --text         送信するテキスト
# --sender       送信元識別子（任意文字列）
# --interval-ms  送信間隔（ミリ秒）
# --count        送信回数（0 の場合は無限送信）

import argparse
import time
from pathlib import Path

import capnp  # type: ignore[import-not-found]


# log_capnpをcapnp.loadでロード
LOG_CAPNP_PATH = Path(__file__).resolve().parent / "cereal" / "log.capnp"
log_capnp = capnp.load(str(LOG_CAPNP_PATH))

import cereal.messaging as messaging

messaging.context = messaging.Context()

CUSTOM_CAPNP_PATH = Path(__file__).resolve().parent / "cereal" / "custom.capnp"
custom_capnp = capnp.load(str(CUSTOM_CAPNP_PATH))


def build_payload(sequence: int, sender: str, text: str) -> bytes:
    msg = custom_capnp.DebugPublishMessage.new_message()
    msg.sequence = sequence
    msg.sender = sender
    msg.text = text
    msg.wallTimeNanos = time.time_ns()
    return msg.to_bytes()

def build_e2eoutput_payload(sequence: int, sender: str, text: str):
    msg = custom_capnp.E2EOutput.new_message()
    msg.aEgo = 0.0
    msg.vEgo = 0.0
    msg.steeringAngleDeg = 0.0
    msg.timestamp = time.time_ns()
    msg.isValid = True
    msg.vEgoPlans = [0.0] * 10
    return msg

def publish_messages(service: str, text: str, sender: str, interval_ms: int, count: int):

    pm = messaging.PubMaster([service])
    print(f"Publishing to {service}. Press Ctrl+C to stop.")

    # log.Eventのスキーマ取得
    event_schema = log_capnp.Event.schema
    fields = event_schema.fields
    print(f"[DEBUG] event_schema.fields:", fields)
    field_obj = fields.get(service) if isinstance(fields, dict) else None
    if field_obj is not None:
        if hasattr(field_obj, "slot"):
            field_type = field_obj.slot.type.which()
        else:
            field_type = field_obj.proto.slot.type.which()
        print(f"[DEBUG] field_type: {field_type} (type: {type(field_type)})")
        # field_typeの値を自動判別してdata/struct型に対応
        is_data = False
        is_struct = False
        if isinstance(field_type, str):
            is_data = field_type.lower() == "data"
            is_struct = field_type.lower() == "struct"
        elif isinstance(field_type, int):
            is_data = field_type == 0
            is_struct = field_type == 11
        else:
            is_data = "data" in str(field_type).lower()
            is_struct = "struct" in str(field_type).lower()
    else:
        raise ValueError(f"service '{service}' not found in log_capnp.Event schema fields")

    sequence = 0
    while count <= 0 or sequence < count:
        if is_data:
            payload = build_payload(sequence=sequence, sender=sender, text=text)
            dat = messaging.new_message(service, size=len(payload), valid=True)
            setattr(dat, service, payload)
            # 送信前にデコードしてprint
            try:
                with custom_capnp.DebugPublishMessage.from_bytes(payload) as msg:
                    print({
                        "sequence": msg.sequence,
                        "sender": msg.sender,
                        "text": msg.text,
                        "wallTimeNanos": msg.wallTimeNanos,
                        "bytes": len(payload),
                    })
            except Exception as e:
                print(f"Failed to decode DebugPublishMessage: {e}")
        elif is_struct:
            payload = build_e2eoutput_payload(sequence, sender, text)
            dat = messaging.new_message(service, valid=True)
            setattr(dat, service, payload)
            # 送信前にデコードしてprint
            try:
                payload_bytes = payload.to_bytes()
                with custom_capnp.E2EOutput.from_bytes(payload_bytes) as msg:
                    print({
                        "aEgo": msg.aEgo,
                        "vEgo": msg.vEgo,
                        "steeringAngleDeg": msg.steeringAngleDeg,
                        "timestamp": msg.timestamp,
                        "isValid": msg.isValid,
                        "vEgoPlans": list(msg.vEgoPlans),
                        "bytes": len(payload_bytes),
                    })
            except Exception as e:
                print(f"Failed to decode E2EOutput: {e}")
        else:
            raise TypeError(f"Unsupported capnp field type: {field_type}")
        pm.send(service, dat)
        sequence += 1
        if interval_ms > 0:
            time.sleep(interval_ms / 1000.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service",
        default="e2eOutput",
        choices=[
            "customReservedRawData1",
            "customReservedRawData2",
            "e2eOutput"
        ],
        help="Target custom raw data service",
    )
    parser.add_argument("--text", default="hello from debug publisher")
    parser.add_argument("--sender", default="debug_publish_message.py")
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of messages to publish. 0 means infinite.",
    )
    args = parser.parse_args()

    try:
        publish_messages(
            service=args.service,
            text=args.text,
            sender=args.sender,
            interval_ms=args.interval_ms,
            count=args.count,
        )
    except KeyboardInterrupt:
        print("Stopped by user.")

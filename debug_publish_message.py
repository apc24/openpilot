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
import zmq

import capnp  # type: ignore[import-not-found]

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


def publish_messages(
    service: str,
    text: str,
    sender: str,
    interval_ms: int,
    count: int,
):
    pm = messaging.PubMaster([service])
    print(f"Publishing to {service}. Press Ctrl+C to stop.")

    sequence = 0
    while count <= 0 or sequence < count:
        # payload = build_payload(sequence=sequence, sender=sender, text=text)

        msg = custom_capnp.E2EOutput.new_message()
        msg.aEgo = 0.0
        msg.vEgo = 0.0
        msg.steeringAngleDeg = 0.0
        msg.timestamp = time.time_ns()
        msg.isValid = True
        msg.vEgoPlans = [0.0] * 10

        dat = messaging.new_message(service, valid=True)
        setattr(dat, service, msg)
        pm.send(service, dat)

        print(
            f"sent: service={service} sequence={sequence} text='{text}' bytes={len(msg.to_bytes())}"
        )
        sequence += 1

        if interval_ms > 0:
            time.sleep(interval_ms / 1000.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service",
        default="e2eOutput",
        choices=[
            "customReservedRawData0",
            "customReservedRawData1",
            "customReservedRawData2",
            "e2eOutput",
        ],
        help="Target custom raw data service",
    )
    parser.add_argument("--addr", default="192.168.1.51", help="Publish target address (IP)")
    parser.add_argument("--text", default="hello from debug publisher")
    parser.add_argument("--sender", default="debug_publish_message.py")
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of messages to publish. 0 means infinite.",
    )
    parser.add_argument("--zmq", action="store_true", help="Use pure pyzmq for sending")
    args = parser.parse_args()

    if args.zmq:
        # --- pyzmqのみで送信 ---
        import zmq
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.connect(f"tcp://{args.addr}:8061")
        print(f"ZMQ mode: sending to tcp://{args.addr}:8061")
        sequence = 0
        try:
            while args.count <= 0 or sequence < args.count:
                msg = custom_capnp.E2EOutput.new_message()
                msg.aEgo = 0.0
                msg.vEgo = 0.0
                msg.steeringAngleDeg = 0.0
                msg.timestamp = time.time_ns()
                msg.isValid = True
                msg.vEgoPlans = [0.0] * 10
                data = msg.to_bytes()
                socket.send(data)
                print(f"sent (zmq): sequence={sequence} bytes={len(data)}")
                sequence += 1
                if args.interval_ms > 0:
                    time.sleep(args.interval_ms / 1000.0)
        except KeyboardInterrupt:
            print("Stopped by user.")
    else:
        pass
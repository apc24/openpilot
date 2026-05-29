# e2eoutput_subscriber.py
# e2eOutputサービスのメッセージを受信し、内容をデコードして表示するサンプル

import capnp
import cereal.messaging as messaging
from pathlib import Path

# custom.capnpのロード
CUSTOM_CAPNP_PATH = Path(__file__).resolve().parent / "cereal" / "custom.capnp"
custom_capnp = capnp.load(str(CUSTOM_CAPNP_PATH))


def subscribe_e2eoutput(addr: str = "192.168.1.51"):
    import cereal.services as services
    service = "e2eOutput"
    port = services.SERVICE_LIST[service].port
    print(f"waiting for {service} messages on {addr} (port {port}) ...")
    sm = messaging.SubMaster([service], addr=addr)
    while True:
        sm.update(1000)
        if sm.updated[service]:
            raw = sm[service]
            try:
                if isinstance(raw, bytes):
                    with custom_capnp.E2EOutput.from_bytes(raw) as msg:
                        print({
                            "aEgo": msg.aEgo,
                            "vEgo": msg.vEgo,
                            "steeringAngleDeg": msg.steeringAngleDeg,
                            "timestamp": msg.timestamp,
                            "isValid": msg.isValid,
                            "vEgoPlans": list(msg.vEgoPlans),
                            "bytes": len(raw),
                        })
                else:
                    # すでにcapnpオブジェクト
                    msg = raw
                    print({
                        "aEgo": msg.aEgo,
                        "vEgo": msg.vEgo,
                        "steeringAngleDeg": msg.steeringAngleDeg,
                        "timestamp": msg.timestamp,
                        "isValid": msg.isValid,
                        "vEgoPlans": list(msg.vEgoPlans),
                        "bytes": None,
                    })
            except Exception as e:
                print(f"Failed to decode E2EOutput: {e}")
                print(raw)
        else:
            print(f"{service} not updated")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="192.168.1.51", help="PublisherのIPアドレス")
    parser.add_argument("--zmq", action="store_true", help="Use pure pyzmq for receiving")
    args = parser.parse_args()

    if args.zmq:
        # --- pyzmqのみで受信 ---
        import zmq
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.bind(f"tcp://{args.addr}:8061")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[zmq] waiting for e2eOutput messages on tcp://{args.addr}:8061 ...")
        while True:
            try:
                raw = socket.recv()
                with custom_capnp.E2EOutput.from_bytes(raw) as msg:
                    print({
                        "aEgo": msg.aEgo,
                        "vEgo": msg.vEgo,
                        "steeringAngleDeg": msg.steeringAngleDeg,
                        "timestamp": msg.timestamp,
                        "isValid": msg.isValid,
                        "vEgoPlans": list(msg.vEgoPlans),
                        "bytes": len(raw),
                    })
            except Exception as e:
                print(f"Failed to decode E2EOutput: {e}")
                print(raw)
    else:
        # --- 既存のmessaging.SubMasterによる受信（コメントアウト可） ---
        # subscribe_e2eoutput(addr=args.addr)
        subscribe_e2eoutput(addr=args.addr)

# e2eoutput_subscriber.py
# e2eOutputサービスのメッセージを受信し、内容をデコードして表示するサンプル

import capnp
import cereal.messaging as messaging
from pathlib import Path

# custom.capnpのロード
CUSTOM_CAPNP_PATH = Path(__file__).resolve().parent / "cereal" / "custom.capnp"
custom_capnp = capnp.load(str(CUSTOM_CAPNP_PATH))


def subscribe_e2eoutput(addr: str = "192.168.1.2"):
    service = "e2eOutput"
    sm = messaging.SubMaster([service], addr=addr)
    print(f"waiting for {service} messages on {addr} ...")
    while True:
        sm.update(1000)
        if sm.updated[service]:
            raw = sm[service]
            raw_bytes = raw if isinstance(raw, bytes) else bytes(raw)
            try:
                with custom_capnp.E2EOutput.from_bytes(raw_bytes) as msg:
                    print({
                        "aEgo": msg.aEgo,
                        "vEgo": msg.vEgo,
                        "steeringAngleDeg": msg.steeringAngleDeg,
                        "timestamp": msg.timestamp,
                        "isValid": msg.isValid,
                        "vEgoPlans": list(msg.vEgoPlans),
                        "bytes": len(raw_bytes),
                    })
            except Exception as e:
                print(f"Failed to decode E2EOutput: {e}")
                print(raw)
        else:
            print(f"{service} not updated")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="192.168.1.2", help="PublisherのIPアドレス")
    args = parser.parse_args()
    subscribe_e2eoutput(addr=args.addr)

# このスクリプトを実行する前に openpilot および metadrive を ZMQ=1 で起動してください。
# また、FW のメッセージ配信ポートを事前に開放してください。
# 例: ZMQ=1 tools/sim/launch_openpilot.sh と ZMQ=1 tools/sim/run_bridge.py
#
# 機能:
# 1) --service で指定したサービスを購読して内容を表示します。
#    例: ZMQ=1 poetry run python debug_subscribe_message.py --addr <IP_ADDRESS> --service customReservedRawData0
#
# 2) roadEncodeData / wideEncodeData のデコード保存モード。
#    --decode-road を付けるとエンコード済みデータをデコードし、PNG で保存します。
#    例: ZMQ=1 poetry run python debug_subscribe_message.py --addr <IP_ADDRESS> --service roadEncodeData --decode-road --frames 10 --out-dir decoded_road_frames
#
# 補足:
# - customReservedRawData* を購読した場合は custom.capnp の DebugPublishMessage としてデコードを試みます。
# - それ以外のサービスは受信した内容をそのまま表示します。


import argparse
import os
import tempfile
from pathlib import Path

import capnp  # type: ignore[import-not-found]

import cereal.messaging as messaging
from cereal import log
from cereal.visionipc import VisionIpcClient, VisionStreamType

messaging.context = messaging.Context()

CUSTOM_CAPNP_PATH = Path(__file__).resolve().parent / "cereal" / "custom.capnp"
custom_capnp = capnp.load(str(CUSTOM_CAPNP_PATH))

V4L2_BUF_FLAG_KEYFRAME = 8


def _codec_name_for_encode_type(encode_type: int) -> str:
    if encode_type == log.EncodeIndex.Type.bigBoxLossless:
        return "ffvhuff"
    if encode_type in (
        log.EncodeIndex.Type.qcameraH264,
        log.EncodeIndex.Type.livestreamH264,
    ):
        return "h264"
    return "hevc"


def _codec_candidates_for_encode_type(encode_type: int) -> list[str]:
    primary = _codec_name_for_encode_type(encode_type)
    candidates = [primary]
    for fallback in ("ffvhuff", "h264", "hevc"):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _save_frame_as_png(frame, out_path: str) -> None:
    frame.to_image().save(out_path, format="PNG")


def _decode_ffvhuff_packet_via_avi(av_module, payload: bytes, width: int, height: int):
    fd, tmp_path = tempfile.mkstemp(suffix=".avi")
    os.close(fd)
    try:
        out = av_module.open(tmp_path, mode="w", format="avi")
        stream = out.add_stream("ffvhuff", rate=20)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        packet = av_module.packet.Packet(payload)
        packet.pts = 0
        packet.dts = 0
        packet.stream = stream
        out.mux(packet)
        out.close()

        inp = av_module.open(tmp_path)
        try:
            for demux_packet in inp.demux(video=0):
                for frame in demux_packet.decode():
                    return frame
        finally:
            inp.close()
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def subscribe_messages(addr: str, service: str = "roadEncodeData"):
    services = [service]
    sm = messaging.SubMaster(services, addr=addr)
    print("waiting...")

    while True:
        sm.update(1000)
        for s in services:
            if sm.updated[s]:
                print("=" * 30, s, sm.logMonoTime[s])
                if s.startswith("customReservedRawData"):
                    raw = sm[s]
                    raw_bytes = raw if isinstance(raw, bytes) else bytes(raw)
                    try:
                        with custom_capnp.DebugPublishMessage.from_bytes(
                            raw_bytes
                        ) as msg:
                            print(
                                {
                                    "sequence": msg.sequence,
                                    "sender": msg.sender,
                                    "text": msg.text,
                                    "wallTimeNanos": msg.wallTimeNanos,
                                    "bytes": len(raw_bytes),
                                }
                            )
                    except Exception as e:
                        print(f"Failed to decode DebugPublishMessage: {e}")
                        print(raw)
                elif s == "e2eOutput":
                    raw = sm[s]
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
                elif s.startswith("roadEncodeData") or s.startswith("wideEncodeData"):
                    msg = sm[s]
                    print(
                        "type:",
                        msg.idx.type,
                        "frame_id:",
                        msg.idx.frameId,
                        "encode_id:",
                        msg.idx.encodeId,
                        "size:",
                        msg.width,
                        "x",
                        msg.height,
                        "header_bytes:",
                        len(msg.header),
                        "payload_bytes:",
                        len(msg.data),
                        "flags:",
                        msg.idx.flags,
                    )
                else:
                    print(sm[s])
            else:
                print(f"{s} not updated")


def decode_road_encode_data(
    addr: str,
    out_dir: str,
    frames: int = 3,
    timeout_ms: int = 1000,
    max_empty: int = 50,
    max_decode_errors: int = 100,
    service: str = "roadEncodeData",
) -> None:
    try:
        import av as av_module  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("PyAV is not installed. Install with: poetry install --with dev")
        return

    sm = messaging.SubMaster([service], addr=addr)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Decoding {service} and saving {frames} frame(s) as .png into: {out_dir}")

    saved = 0
    empty = 0
    decode_errors = 0
    current_encode_type = None
    codec_candidates: list[str] = []
    codec_candidate_idx = 0
    codec = None
    seen_iframe = False

    while saved < frames and empty < max_empty:
        sm.update(timeout_ms)
        if not sm.updated[service]:
            empty += 1
            print(f"{service} not updated ({empty}/{max_empty})")
            continue

        msg = sm[service]
        idx = msg.idx
        encode_type = idx.type

        if encode_type != current_encode_type or codec is None:
            current_encode_type = encode_type
            codec_candidates = _codec_candidates_for_encode_type(encode_type)
            codec_candidate_idx = 0
            codec = av_module.CodecContext.create(
                codec_candidates[codec_candidate_idx], "r"
            )
            if (
                codec.name == "ffvhuff"
                or encode_type == log.EncodeIndex.Type.bigBoxLossless
            ):
                codec.width = msg.width
                codec.height = msg.height
                codec.pix_fmt = "yuv420p"
            seen_iframe = False
            print(
                "switched codec:",
                codec.name,
                "encode_type:",
                encode_type,
                "candidates:",
                codec_candidates,
            )

        if not seen_iframe and not (idx.flags & V4L2_BUF_FLAG_KEYFRAME):
            print("waiting for keyframe, flags:", idx.flags)
            continue

        should_apply_header = len(msg.header) > 0 and (
            not seen_iframe or (idx.flags & V4L2_BUF_FLAG_KEYFRAME)
        )
        if should_apply_header:
            try:
                if codec.name == "ffvhuff":
                    codec.extradata = bytes(msg.header)
                else:
                    codec.decode(av_module.packet.Packet(msg.header))
            except av_module.error.FFmpegError as e:
                print("failed to decode header packet:", e)
                decode_errors += 1
                if codec_candidate_idx + 1 < len(codec_candidates):
                    codec_candidate_idx += 1
                codec = av_module.CodecContext.create(
                    codec_candidates[codec_candidate_idx], "r"
                )
                if (
                    codec.name == "ffvhuff"
                    or encode_type == log.EncodeIndex.Type.bigBoxLossless
                ):
                    codec.width = msg.width
                    codec.height = msg.height
                    codec.pix_fmt = "yuv420p"
                print("header fallback codec:", codec.name)
                seen_iframe = False
                if decode_errors >= max_decode_errors:
                    print(f"Too many decode errors ({decode_errors}). Aborting.")
                    break
                continue
            seen_iframe = True
        elif not seen_iframe:
            seen_iframe = True

        try:
            if (
                encode_type == log.EncodeIndex.Type.bigBoxLossless
                and codec.name == "ffvhuff"
            ):
                frame = _decode_ffvhuff_packet_via_avi(
                    av_module, bytes(msg.data), msg.width, msg.height
                )
                decoded_frames = [frame] if frame is not None else []
            else:
                decoded_frames = codec.decode(av_module.packet.Packet(msg.data))
        except av_module.error.FFmpegError as e:
            print("failed to decode payload packet:", e)
            decode_errors += 1
            if codec_candidate_idx + 1 < len(codec_candidates):
                codec_candidate_idx += 1
            codec = av_module.CodecContext.create(
                codec_candidates[codec_candidate_idx], "r"
            )
            if (
                codec.name == "ffvhuff"
                or encode_type == log.EncodeIndex.Type.bigBoxLossless
            ):
                codec.width = msg.width
                codec.height = msg.height
                codec.pix_fmt = "yuv420p"
            print("payload fallback codec:", codec.name)
            seen_iframe = False
            if decode_errors >= max_decode_errors:
                print(f"Too many decode errors ({decode_errors}). Aborting.")
                break
            continue

        if len(decoded_frames) == 0:
            print("packet decoded but no complete frame yet")
            continue

        frame = decoded_frames[-1]
        out_path = os.path.join(
            out_dir,
            f"road_{saved:03d}_frame{idx.frameId}_enc{idx.encodeId}_{codec.name}_{frame.width}x{frame.height}.png",
        )
        _save_frame_as_png(frame, out_path)

        print(
            "saved:",
            out_path,
            "type:",
            encode_type,
            "bytes:",
            len(msg.data),
            "timestamp_ns:",
            msg.unixTimestampNanos,
        )
        saved += 1
        empty = 0
        decode_errors = 0

    if saved == 0:
        print(
            "No decodable frame was saved. Check that encoded stream is being published and includes keyframes."
        )
    else:
        print(f"Done. Saved {saved} frame(s) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", help="Address to subscribe to", default="127.0.0.1")
    parser.add_argument(
        "--service",
        default="roadEncodeData",
        help="Service name used when --target messages",
    )
    parser.add_argument(
        "--frames", type=int, default=10, help="Number of camera frames to print"
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=1000,
        help="Per-frame recv timeout in ms for camera mode",
    )
    parser.add_argument(
        "--max-empty",
        type=int,
        default=30,
        help="Max consecutive empty recv attempts for camera mode",
    )
    parser.add_argument(
        "--max-decode-errors",
        type=int,
        default=100,
        help="Max consecutive decode errors before aborting in --decode-road mode",
    )
    parser.add_argument(
        "--decode-road",
        action="store_true",
        help="Decode roadEncodeData packets and save sample frames",
    )
    parser.add_argument(
        "--out-dir",
        default="decoded_road_frames",
        help="Output directory for decoded road frames",
    )
    args = parser.parse_args()
    if args.addr is None:
        print("Please provide an address to subscribe to.")
    else:
        if (
            args.service == "roadEncodeData" or args.service == "wideEncodeData"
        ) and args.decode_road:
            decode_road_encode_data(
                args.addr,
                out_dir=args.out_dir,
                frames=args.frames,
                timeout_ms=args.timeout_ms,
                max_empty=args.max_empty,
                max_decode_errors=args.max_decode_errors,
                service=args.service,
            )
        else:
            subscribe_messages(args.addr, service=args.service)

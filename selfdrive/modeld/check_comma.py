#!/usr/bin/env python3
import argparse
import time

from cereal.messaging import SubMaster

try:
  from openpilot.selfdrive.modeld.constants import ModelConstants
  MODEL_FREQ = float(ModelConstants.MODEL_FREQ)
except Exception:
  MODEL_FREQ = 20.0


def evaluate_e2e_state(sm: SubMaster):
  if sm.recv_frame['e2eOutput'] < 0:
    return "none", "no_e2eOutput_received", None

  e2e = sm['e2eOutput']
  age = (sm.frame - sm.recv_frame['e2eOutput']) / MODEL_FREQ

  if age > 0.5:
    return "none", "stale", age
  if not sm.valid['e2eOutput']:
    return "none", "sm_invalid", age
  if not e2e.isValid:
    return "none", "msg_invalid", age

  return "ok", "usable", age


def main(addr: str, timeout_ms: int, print_every: int):
  sm = SubMaster(["modelV2", "e2eOutput", "carState"], addr=addr)
  counter = 0

  print(f"[check] subscribe addr={addr} model_freq={MODEL_FREQ}")

  while True:
    sm.update(timeout_ms)
    counter += 1

    if not sm.updated['modelV2'] and (counter % print_every != 0):
      continue

    state, reason, age = evaluate_e2e_state(sm)

    use_e2e = None
    desired_curvature = None
    if sm.updated['modelV2'] or sm.recv_frame['modelV2'] >= 0:
      try:
        use_e2e = bool(sm['modelV2'].action.useE2eOutput)
        desired_curvature = float(sm['modelV2'].action.desiredCurvature)
      except Exception:
        pass

    if sm.recv_frame['e2eOutput'] >= 0:
      e2e = sm['e2eOutput']
      e2e_info = (
        f"e2e(vEgo={float(e2e.vEgo):.3f}, "
        f"aEgo={float(e2e.aEgo):.3f}, "
        f"steerDeg={float(e2e.steeringAngleDeg):.3f}, "
        f"isValid={bool(e2e.isValid)})"
      )
    else:
      e2e_info = "e2e(unreceived)"

    age_str = "n/a" if age is None else f"{age:.3f}s"
    print(
      "[check] "
      f"state={state} reason={reason} age={age_str} "
      f"modelV2.useE2eOutput={use_e2e} "
      f"desiredCurvature={desired_curvature} "
      f"{e2e_info}"
    )

    time.sleep(0.01)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--addr", default="192.168.1.51", help="SubMaster addr")
  parser.add_argument("--timeout-ms", type=int, default=1000, help="SubMaster update timeout")
  parser.add_argument("--print-every", type=int, default=20, help="Print heartbeat interval when modelV2 is not updated")
  args = parser.parse_args()
  main(args.addr, args.timeout_ms, args.print_every)

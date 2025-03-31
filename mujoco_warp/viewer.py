# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""An example integration of MJWarp with the MuJoCo viewer."""

import logging
import time
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np
import warp as wp
from absl import app
from absl import flags

import mujoco_warp as mjwarp

_MODEL_PATH = flags.DEFINE_string(
  "mjcf", None, "Path to a MuJoCo MJCF file.", required=True
)
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool(
  "clear_kernel_cache", False, "Clear kernel cache (to calculate full JIT time)"
)
_ENGINE = flags.DEFINE_enum("engine", "mjwarp", ["mjwarp", "mjc"], "Simulation engine")
_LS_PARALLEL = flags.DEFINE_bool(
  "ls_parallel", False, "Engine solver with parallel linesearch"
)
_VIEWER_GLOBAL_STATE = {
  "running": True,
  "step_once": False,
}


def key_callback(key: int) -> None:
  if key == 32:  # Space bar
    _VIEWER_GLOBAL_STATE["running"] = not _VIEWER_GLOBAL_STATE["running"]
    logging.info("RUNNING = %s", _VIEWER_GLOBAL_STATE["running"])
  elif key == 46:  # period
    _VIEWER_GLOBAL_STATE["step_once"] = True


def _main(argv: Sequence[str]) -> None:
  """Launches MuJoCo passive viewer fed by MJWarp."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(f"Loading model from: {_MODEL_PATH.value}.")
  if _MODEL_PATH.value.endswith(".mjb"):
    mjm = mujoco.MjModel.from_binary_path(_MODEL_PATH.value)
  else:
    mjm = mujoco.MjModel.from_xml_path(_MODEL_PATH.value)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)

  if _ENGINE.value == "mjc":
    print("Engine: MuJoCo C")
  else:  # mjwarp
    print("Engine: MuJoCo Warp")
    m = mjwarp.put_model(mjm)
    m.opt.ls_parallel = _LS_PARALLEL.value
    d = mjwarp.put_data(mjm, mjd)

    if _CLEAR_KERNEL_CACHE.value:
      wp.clear_kernel_cache()

    start = time.time()
    print("Compiling the model physics step...")
    mjwarp.step(m, d)
    # double warmup to work around issues with compilation during graph capture:
    mjwarp.step(m, d)
    # capture the whole step function as a CUDA graph
    with wp.ScopedCapture() as capture:
      mjwarp.step(m, d)
    graph = capture.graph
    elapsed = time.time() - start
    print(f"Compilation took {elapsed}s.")

  viewer = mujoco.viewer.launch_passive(mjm, mjd, key_callback=key_callback)
  with viewer:
    while True:
      start = time.time()

      if _ENGINE.value == "mjc":
        mujoco.mj_step(mjm, mjd)
      else:  # mjwarp
        # TODO(robotics-simulation): recompile when changing disable flags, etc.
        wp.copy(d.ctrl, wp.array([mjd.ctrl.astype(np.float32)]))
        wp.copy(d.act, wp.array([mjd.act.astype(np.float32)]))
        wp.copy(d.xfrc_applied, wp.array([mjd.xfrc_applied.astype(np.float32)]))
        wp.copy(d.qpos, wp.array([mjd.qpos.astype(np.float32)]))
        wp.copy(d.qvel, wp.array([mjd.qvel.astype(np.float32)]))
        d.time = mjd.time

        if _VIEWER_GLOBAL_STATE["running"]:
          wp.capture_launch(graph)
          wp.synchronize()
        elif _VIEWER_GLOBAL_STATE["step_once"]:
          _VIEWER_GLOBAL_STATE["step_once"] = False
          wp.capture_launch(graph)
          wp.synchronize()

        mjwarp.get_data_into(mjd, mjm, d)

      viewer.sync()

      elapsed = time.time() - start
      if elapsed < mjm.opt.timestep:
        time.sleep(mjm.opt.timestep - elapsed)


def main():
  app.run(_main)


if __name__ == "__main__":
  main()

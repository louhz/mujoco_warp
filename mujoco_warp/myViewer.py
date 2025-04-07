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
import os
import mujoco_warp as mjwarp



  
parent_dir = "/home/haozhe/Dropbox/physics/_data/allegro/wonik_allegro"
xml_file = os.path.join(parent_dir, "scene_bluelegotest.xml")






_MODEL_PATH = flags.DEFINE_string(
  "mjcf", None, "Path to a MuJoCo MJCF file.", required=True
)
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool(
  "clear_kernel_cache", False, "Clear kernel cache (to calculate full JIT time)"
)
_ENGINE = flags.DEFINE_enum("engine", "mjc", ["mjwarp", "mjc"], "Simulation engine")
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

def set_joint_angles(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, joint_names, angles):
    """
    Sets the angles (in radians) for a given list of joint names.
    'angles' should match the length of 'joint_names'.
    """
    for jname, angle in zip(joint_names, angles):
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if joint_id < 0:
            raise ValueError(f"Joint '{jname}' not found in model.")

        qpos_adr = mj_model.jnt_qposadr[joint_id]
        mj_data.qpos[qpos_adr] = angle

    mujoco.mj_forward(mj_model, mj_data)

def _main(argv: Sequence[str]) -> None:
  """Launches MuJoCo passive viewer fed by MJWarp."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(f"Loading model from: {_MODEL_PATH.value}.")


  mjm = mujoco.MjModel.from_xml_path(_MODEL_PATH.value)
  mjm.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
  # mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  is_sparse = False
  mjm.opt.jacobian = is_sparse
  mjm.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  print("Number of actuators (mj_model.nu):", mjm.nu)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  hand_joint_names = [
        "ffj0",
        "ffj1",
        "ffj2",
        "ffj3",
        "mfj0",
        "mfj1",
        "mfj2",
        "mfj3",
        "rfj0",
        "rfj1",
        "rfj2",
        "rfj3",
        "thj0",
        "thj1",
        "thj2",
        "thj3",
    ]
    # Initialize those joints to some angle (optional)
  init_hand_angles = [0.0] * 16  # in radians
  set_joint_angles(mjm, mjd, hand_joint_names, init_hand_angles)
  # mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  raw_action = np.array([
        0.18946200609207153,
        1.336256742477417,
        0.7085919976234436,
        0.13329032063484192,
        0.07188903540372849,
        1.5545207262039185,
        0.6563715934753418,
        0.3384369909763336,
        -0.30586832761764526,
        1.4823938608169556,
        0.6193418502807617,
        0.6304949522018433,
        1.4147015810012817,
        0.6929757595062256,
        0.7424201369285583,
        0.5614045858383179,
    ])


  if _ENGINE.value == "mjc":
    print("Engine: MuJoCo C")
  else:  # mjwarp
    print("Engine: MuJoCo Warp")
    m = mjwarp.put_model(mjm)
    m.opt.ls_parallel = _LS_PARALLEL.value
    d = mjwarp.put_data(mjm, mjd)
    d.ctrl.numpy()[0][:] = init_hand_angles
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
        mjd.ctrl[:] = raw_action
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
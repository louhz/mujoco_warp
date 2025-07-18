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

import warp as wp

collect_ignore = ["benchmark/mujoco_menagerie"]


def pytest_addoption(parser):
  parser.addoption("--cpu", action="store_true", default=False, help="run tests with cpu")
  parser.addoption(
    "--verify_cuda",
    action="store_true",
    default=False,
    help="run tests with cuda error checking",
  )
  parser.addoption("--lineinfo", action="store_true", default=False, help="add lineinfo to warp kernel")


def pytest_configure(config):
  if config.getoption("--cpu"):
    wp.set_device("cpu")
  if config.getoption("--verify_cuda"):
    wp.config.verify_cuda = True
  if config.getoption("--lineinfo"):
    wp.config.lineinfo = True

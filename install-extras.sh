#!/bin/bash
# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Installs meltingpot extras on Linux/macOS.

set -euxo pipefail


function check_setup() {
  echo -e "\nChecking meltingpot is installed..."
  python -c 'import meltingpot'
}


function install_extras() {
  echo -e "\nInstalling meltingpot extras..."
  pip install .[rllib,pettingzoo]
}


function test_extras() {
  echo -e "\nTesting meltingpot extras..."
  # Test RLLib and Petting Zoo training scripts.
  # TODO(b/265139141): Add PettingZoo test.
  test_rllib
}


function main() {
  check_setup
  install_extras
}


main "$@"

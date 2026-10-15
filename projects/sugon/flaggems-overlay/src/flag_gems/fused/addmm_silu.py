# Copyright 2026 FlagOS Contributors
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

from flag_gems.ops.addmm import addmm
from flag_gems.ops.silu import silu


def addmm_silu(bias, mat1, mat2, *, beta=1, alpha=1):
    """Apply SiLU to AddMM using FlagGems-owned Triton computation."""

    return silu(addmm(bias, mat1, mat2, beta=beta, alpha=alpha))

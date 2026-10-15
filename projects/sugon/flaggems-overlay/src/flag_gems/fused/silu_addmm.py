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


def silu_addmm(input, bias, mat2, *, beta=1, alpha=1):
    """Apply SiLU followed by AddMM using FlagGems-owned Triton kernels."""

    return addmm(bias, silu(input), mat2, beta=beta, alpha=alpha)

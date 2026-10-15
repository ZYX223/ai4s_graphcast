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

from flag_gems.ops.add import add
from flag_gems.ops.addmm import addmm
from flag_gems.ops.silu import silu


def add_add_silu_addmm(
    addend,
    residual_a,
    residual_b,
    bias,
    mat2,
    *,
    beta=1,
    alpha=1,
):
    """Apply two ordered adds, SiLU, and AddMM with FlagGems kernels."""

    activated = silu(add(addend, residual_a), residual_b)
    return addmm(bias, activated, mat2, beta=beta, alpha=alpha)

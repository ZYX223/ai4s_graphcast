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
from flag_gems.ops.layernorm import layer_norm


def post_layer_norm_residual(
    x, residual, normalized_shape, weight=None, bias=None, eps=1e-5
):
    """Apply LayerNorm then residual add using FlagGems-owned kernels."""

    normalized, _, _ = layer_norm(x, normalized_shape, weight, bias, eps)
    return add(normalized, residual)


__all__ = ["post_layer_norm_residual"]

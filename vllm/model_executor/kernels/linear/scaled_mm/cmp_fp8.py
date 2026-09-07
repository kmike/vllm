# SPDX-License-Identifier: Apache-2.0
# cmp_fp8: drop-in FP8 GEMM for CMP 170HX (sm_80, 70 SMs).
# Step 2 skeleton (Sep 8): subclasses MarlinFP8 and delegates — proves
# registry wiring + oracle diffing with zero perf claim. The fused
# dequant+HMMA core lands here in Step 3 (Tier-1 harness first).
# Selection seam: init_fp8_linear_kernel(..., force_kernel=CmpFp8ScaledMMLinearKernel).

import torch

from .marlin import MarlinFP8ScaledMMLinearKernel


class CmpFp8ScaledMMLinearKernel(MarlinFP8ScaledMMLinearKernel):
    """
    FP8 Marlin-compatible kernel, 70-SM tuned (CMP 170HX).

    Skeleton: identical math to MarlinFP8 (super().apply_weights), so the
    oracle diff must be bit-exact. Step 3 overrides apply_weights with the
    fused core; the repack path (process_weights_after_loading) stays
    Marlin-format unless the core needs its own layout (repack-at-load
    allowed: bit-exact layout shuffle, no requant).
    """

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().apply_weights(layer, x, bias)

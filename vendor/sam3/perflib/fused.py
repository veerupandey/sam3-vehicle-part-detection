# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch

addmm_act_op = torch.ops.aten._addmm_activation


def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled() or mat1.device.type != "cuda":
        # Fallback (single-GPU / non-reentrant act-ckpt): standard linear+act.
        act = activation() if isinstance(activation, type) else activation
        return act(torch.nn.functional.linear(mat1, linear.weight, linear.bias))
    self = linear.bias.detach()
    mat2 = linear.weight.detach()
    orig_dtype = mat1.dtype
    self = self.to(torch.bfloat16)
    mat1 = mat1.to(torch.bfloat16)
    mat2 = mat2.to(torch.bfloat16)
    mat1_flat = mat1.view(-1, mat1.shape[-1])
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
        return y.view(mat1.shape[:-1] + (y.shape[-1],)).to(orig_dtype)
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
        return y.view(mat1.shape[:-1] + (y.shape[-1],)).to(orig_dtype)
    raise ValueError(f"Unexpected activation {activation}")

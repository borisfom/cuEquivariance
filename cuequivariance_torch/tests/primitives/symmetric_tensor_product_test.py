# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch

import cuequivariance as cue
import cuequivariance.segmented_tensor_product as stp
import cuequivariance_torch as cuet
from cuequivariance import descriptors

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def make_descriptors():
    yield descriptors.symmetric_contraction(
        cue.Irreps("SO3", "0 + 1 + 2"), cue.Irreps("SO3", "0"), [3]
    ).ds
    yield descriptors.symmetric_contraction(
        cue.Irreps("O3", "0e + 1o + 2e"), cue.Irreps("O3", "0e + 1o"), [4]
    ).ds
    yield descriptors.symmetric_contraction(
        cue.Irreps("SU2", "0 + 1/2"), cue.Irreps("SU2", "0 + 1/2"), [5]
    ).ds

    d1 = stp.SegmentedTensorProduct.from_subscripts(",,")
    d1.add_path(None, None, None, c=2.0)

    d3 = stp.SegmentedTensorProduct.from_subscripts(",,,,")
    d3.add_path(None, None, None, None, None, c=3.0)

    yield [d1, d3]


settings1 = [
    (torch.float64, torch.float64, 1e-12),
    (torch.float32, torch.float32, 1e-5),
    (torch.float32, torch.float64, 1e-5),
]

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    settings1 += [
        (torch.float16, torch.float32, 1.0),
        (torch.float16, torch.float64, 0.1),
        (torch.bfloat16, torch.float32, 1.0),
        (torch.bfloat16, torch.float64, 0.5),
    ]


@pytest.mark.parametrize("ds", make_descriptors())
@pytest.mark.parametrize("dtype, math_dtype, tol", settings1)
@pytest.mark.parametrize("use_fallback", [False, True])
def test_primitive_indexed_symmetric_tensor_product_cuda_vs_fx(
    ds: list[stp.SegmentedTensorProduct],
    dtype,
    math_dtype,
    tol: float,
    use_fallback: bool,
):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    m = cuet.IWeightedSymmetricTensorProduct(
        ds, math_dtype=math_dtype, device=device, use_fallback=use_fallback
    )
    if use_fallback is False:
        m = torch.jit.script(m)

    x0 = torch.randn((2, m.x0_size), device=device, dtype=dtype, requires_grad=True)
    i0 = torch.tensor([0, 1, 0], dtype=torch.int32, device=device)
    x1 = torch.randn(
        (i0.size(0), m.x1_size), device=device, dtype=dtype, requires_grad=True
    )
    x0_ = x0.clone().to(torch.float64)
    x1_ = x1.clone().to(torch.float64)

    out1 = m(x0, i0, x1)
    m = cuet.IWeightedSymmetricTensorProduct(
        ds,
        math_dtype=torch.float64,
        device=device,
        use_fallback=True,
        optimize_fallback=True,
    )
    out2 = m(x0_, i0, x1_)

    assert out1.dtype == dtype

    torch.testing.assert_close(out1, out2.to(dtype), atol=tol, rtol=tol)

    grad1 = torch.autograd.grad(out1.sum(), (x0, x1), create_graph=True)
    grad2 = torch.autograd.grad(out2.sum(), (x0_, x1_), create_graph=True)

    for g1, g2 in zip(grad1, grad2):
        torch.testing.assert_close(g1, g2.to(dtype), atol=10 * tol, rtol=10 * tol)

    double_grad1 = torch.autograd.grad(sum(g.sum() for g in grad1), (x0, x1))
    double_grad2 = torch.autograd.grad(sum(g.sum() for g in grad2), (x0_, x1_))

    for g1, g2 in zip(double_grad1, double_grad2):
        torch.testing.assert_close(g1, g2.to(dtype), atol=100 * tol, rtol=100 * tol)


settings2 = [
    (torch.float64, torch.float64),
    (torch.float32, torch.float32),
    (torch.float32, torch.float64),
]

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    settings2 += [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ]


@pytest.mark.parametrize("dtype, math_dtype", settings2)
@pytest.mark.parametrize("use_fallback", [False, True])
def test_math_dtype(dtype: torch.dtype, math_dtype: torch.dtype, use_fallback: bool):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    ds = descriptors.symmetric_contraction(
        cue.Irreps("SO3", "0 + 1 + 2"), cue.Irreps("SO3", "0"), [1, 2, 3]
    ).ds
    m = cuet.IWeightedSymmetricTensorProduct(
        ds, math_dtype=math_dtype, device=device, use_fallback=use_fallback
    )
    x0 = torch.randn((20, m.x0_size), dtype=dtype, device=device)
    i0 = torch.randint(0, m.x0_size, (1000,), dtype=torch.int32, device=device)
    x1 = torch.randn((i0.size(0), m.x1_size), dtype=dtype, device=device)

    out1 = m(x0, i0, x1)

    # .to should have no effect
    for param in m.parameters():
        assert False  # no parameters
    m = m.to(torch.float64)

    out2 = m(x0, i0, x1)

    assert out1.dtype == dtype
    assert out2.dtype == dtype
    assert (out1 == out2).all()

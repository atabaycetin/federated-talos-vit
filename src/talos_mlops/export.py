"""Export stage: merge backbone + tau + head into one ONNX graph, then INT8-quantize it.

Sparse masks don't speed up dense CPU kernels; the latency/size win comes from
quantization. `benchmark.py` measures both against the PyTorch model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from talos_mlops import bundle as B
from talos_mlops.model import load_task_model
from talos_mlops.preprocessing import IMAGE_SIZE


def export_onnx(bundle_dir: Path, opset: int = 18, quantize: bool = True) -> dict:
    model = load_task_model(bundle_dir).eval()
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    fp32_path = bundle_dir / B.ONNX_FP32
    torch.onnx.export(
        model,
        (dummy,),
        str(fp32_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=256)}},
        opset_version=opset,
        dynamo=True,
        external_data=False,
        verbose=False,
    )

    # Parity check against PyTorch before anything downstream trusts the graph.
    probe = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        expected = model(probe).numpy()
    got = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"]).run(
        None, {"input": probe.numpy()}
    )[0]
    max_diff = float(np.abs(got - expected).max())
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {max_diff}")
    info = {"onnx_fp32_mb": B.size_mb(fp32_path), "onnx_parity_max_abs_diff": max_diff}

    if quantize:
        int8_path = bundle_dir / B.ONNX_INT8
        with tempfile.TemporaryDirectory() as tmp:
            pre = Path(tmp) / "pre.onnx"
            quant_pre_process(str(fp32_path), str(pre))
            # Dynamic quantization: INT8 weights, activations quantized on the fly. No
            # calibration data needed, and it covers the MatMuls that dominate a ViT.
            quantize_dynamic(str(pre), str(int8_path), weight_type=QuantType.QInt8)
        info["onnx_int8_mb"] = B.size_mb(int8_path)

    B.update_meta(bundle_dir, onnx=info)
    print(f"[export] {info}")
    return info

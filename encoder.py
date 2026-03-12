"""
Converts face embedding models to CoreML for bundling in the iOS library.

Default model: InceptionResNetV1 (VGGFace2, 160x160, 512-dim)
  - Trained on VGGFace2: diverse poses, expressions, backgrounds
  - Robust to angled and partially visible faces — no face alignment required
  - Suitable for general-purpose photo library scanning

Alternative: InsightFace w600k_mbf (ArcFace, WebFace600K, 112x112, 512-dim)
  - Requires aligned frontal faces to work correctly
  - Better for controlled enrollment scenarios (frontal selfies)
  - Set as default: change DEFAULT_MODEL = "arcface" below

Requirements:
    pip install torch facenet-pytorch coremltools
    pip install onnx onnx2torch  # only needed for arcface

Usage:
    python encoder.py
"""

import os
import sys
import coremltools as ct
import torch

DEFAULT_MODEL = "inceptionresnetv1"  # "inceptionresnetv1" or "arcface"


# ── InceptionResNetV1 (default) ───────────────────────────────────────────────

def convert_inceptionresnetv1():
    from facenet_pytorch import InceptionResnetV1

    weights_path = "mobilefacenet.pt"
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found. Download it first.")
        sys.exit(1)

    print("Loading InceptionResNetV1 weights...")
    model = InceptionResnetV1(classify=True, num_classes=8631)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.classify = False  # switch to embedding mode (512-dim output)

    print("Tracing model...")
    example = torch.randn(1, 3, 160, 160)
    traced = torch.jit.trace(model, example)

    print("Converting to CoreML (Float32, neuralnetwork)...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 3, 160, 160))],
        minimum_deployment_target=ct.target.iOS14,
        convert_to="neuralnetwork",
    )
    mlmodel.save("MobileFaceNet.mlpackage")
    print("Saved MobileFaceNet.mlpackage")
    print()
    print("Compile and install with:")
    print("  xcrun coremlc compile MobileFaceNet.mlpackage ios/")


# ── InsightFace w600k_mbf (ArcFace, optional) ─────────────────────────────────

def convert_arcface():
    import urllib.request
    import zipfile
    import onnx
    from onnx2torch import convert as onnx2torch_convert

    zip_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
    zip_path = "buffalo_sc.zip"
    onnx_path = "w600k_mbf.onnx"

    if not os.path.exists(onnx_path):
        print(f"Downloading {zip_path}...")
        urllib.request.urlretrieve(zip_url, zip_path)
        print("Extracting w600k_mbf.onnx...")
        with zipfile.ZipFile(zip_path) as z:
            z.extract(onnx_path)

    print("Loading ONNX model...")
    onnx_model = onnx.load(onnx_path)

    # coremltools 7+ dropped direct ONNX support — convert via PyTorch
    print("Converting ONNX → PyTorch...")
    torch_model = onnx2torch_convert(onnx_model)
    torch_model.eval()
    example = torch.zeros(1, 3, 112, 112)
    traced = torch.jit.trace(torch_model, example)

    print("Converting PyTorch → CoreML (Float32, iOS 18)...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input.1", shape=(1, 3, 112, 112))],
        minimum_deployment_target=ct.target.iOS18,
    )
    mlmodel.save("ArcFace.mlpackage")
    print("Saved ArcFace.mlpackage")
    print()
    print("Compile and install with:")
    print("  xcrun coremlc compile ArcFace.mlpackage ios/")
    print()
    print("When using via setModel(), pass:")
    print("  { modelUri: '<path>/ArcFace.mlmodelc', embeddingSize: 512, inputSize: 112 }")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    if model == "arcface":
        convert_arcface()
    else:
        convert_inceptionresnetv1()

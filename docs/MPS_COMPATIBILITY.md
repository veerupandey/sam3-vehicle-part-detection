# Apple Silicon MPS compatibility

`run_mps.py` is an image-only evaluation path for Apple Silicon. It was tested
with PyTorch 2.6.0 and MPS available.

The adapted source under `vendor/sam3` preserves the trained parameter values.
It makes execution-level changes required by the MPS backend: it avoids
video-only Triton imports, uses real-valued rotary embeddings, replaces a
CUDA-only fused activation with standard PyTorch operations, performs image
resize on CPU before transfer to MPS, and keeps cached positional tensors on
the active device.

The real-valued rotary implementation agrees with the original CPU complex
implementation to a maximum absolute error of `2.38e-7` in a local check.

MPS evaluation runs in float32; the historical evaluation used CUDA bfloat16.
Expect small numerical differences. Run the complete validation set before
using MPS metrics as a release or regression gate.

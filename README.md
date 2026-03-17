# pallas-kernel

JAX/Flax Pallas kernels for Gated Linear Attention (GLA).

## Installation

Install directly from the repository:

```bash
pip install git+https://github.com/primatrix/pallas-kernel.git
```

Or build and install locally:

```bash
git clone https://github.com/primatrix/pallas-kernel.git
cd pallas-kernel
pip install .
```

### Optional dependencies

For GPU support (CUDA 12):

```bash
pip install "pallas-kernel[gpu] @ git+https://github.com/primatrix/pallas-kernel.git"
```

For TPU support:

```bash
pip install "pallas-kernel[tpu] @ git+https://github.com/primatrix/pallas-kernel.git"
```

For development:

```bash
pip install -e ".[dev]"
```

## Building packages

Use the provided build script to create distributable packages:

```bash
./scripts/build.sh        # Build sdist and wheel into dist/
./scripts/build.sh clean  # Remove build artifacts
```

## Usage

```python
from pallas_kernel.ops.gla import chunk_gla, fused_recurrent_gla, fused_chunk_gla
from pallas_kernel.layers.gla import GatedLinearAttention
from pallas_kernel.modules.layernorm import RMSNorm
```

## License

Apache License 2.0

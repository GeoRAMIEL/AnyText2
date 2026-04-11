# AnyText2 Setup Notes

## Target Machine

- **Provider**: vast.ai
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB VRAM, compute capability sm_120)
- **CUDA Driver**: 590.48.01 (CUDA 13.1)
- **OS**: Headless Ubuntu (Linux)

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n anytext2 python=3.10.6
conda activate anytext2
```

### 2. Install PyTorch with Blackwell (sm_120) Support

Standard PyTorch builds (up to cu126) do not support Blackwell GPUs. A nightly build with cu128 is required:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --no-deps
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Expected: 2.11.0+cu128 12.8 True
```

### 3. Install CUDA Runtime Libraries via Conda

PyTorch cu128 needs CUDA 12.x runtime libraries that may not be present on the system:

```bash
conda remove -n anytext2 cudatoolkit --force  # remove old 11.8 pin if present
conda install -n anytext2 cuda-cudart cuda-libraries cuda-cupti -c nvidia/label/cuda-12.6.0
```

### 4. Fix LD_LIBRARY_PATH

The vast.ai container ships a system Python 3.12 with its own torch. Its `LD_LIBRARY_PATH` includes `/usr/local/lib/python3.12/dist-packages/torch/lib`, which leaks incompatible `.so` files into the conda env. Fix by setting env-scoped vars:

```bash
conda env config vars set LD_LIBRARY_PATH=/root/miniconda3/envs/anytext2/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
conda deactivate
conda activate anytext2
```

### 5. Install Dependencies from environment.yaml (with fixes)

Install the pip packages from `environment.yaml`, but with these changes:

| Package | Issue | Fix |
|---|---|---|
| `setuptools` | v82+ removed `pkg_resources`, needed by modelscope | `pip install "setuptools<81"` |
| `albumentations==0.4.3` | Fails to build without `pkg_resources` | Install setuptools first, or use `albumentations==1.3.1` |
| `opencc` | Missing from environment.yaml, needed by `t3_dataset.py` | `pip install opencc-python-reimplemented` |
| `datasets` | Missing transitive dep of modelscope | `pip install "datasets==2.18.0"` |
| `diffusers==0.10.2` | Uses removed `cached_download` from huggingface_hub | `pip install "diffusers==0.25.1"` |
| `huggingface_hub` | Must satisfy both gradio (>=0.25.1) and transformers (<1.0) | `pip install "huggingface_hub==0.25.1"` |
| `tensorflow==2.13.0` | Needed for translation pipeline; GPU version not needed | `pip install "tensorflow-cpu==2.13.0"` |
| `typing-extensions` | TF 2.13 downgrades to 4.5.0, breaking pydantic/gradio | `pip install "typing-extensions>=4.14.1"` |
| `opencv-python` | Not needed on headless server | `pip install opencv-python-headless` (ignore basicsr warning) |
| `xformers==0.0.20` | Incompatible with both new PyTorch and Blackwell GPU | See "Disable xformers" below |
| `gradio==5.12.0` | Bug in `gradio_client` with pydantic schema parsing | `pip install "gradio==5.23.0"` |
| `transformers==4.34.1` | Strict huggingface_hub<1.0 check | `pip install "transformers>=4.40.0,<5"` |

### 6. Disable xformers (Required for Blackwell GPUs)

xformers has no support for compute capability 12.0 (Blackwell). The codebase falls back to vanilla attention, but the VAE attention block does not fall back automatically.

Edit two files:

**`ldm/modules/diffusionmodules/model.py`** (around line 14):
```python
# Replace the try/except block with:
XFORMERS_IS_AVAILBLE = False
```

**`ldm/modules/attention.py`** (same pattern):
```python
XFORMERS_IS_AVAILBLE = False
```

With 96 GB VRAM, vanilla attention works fine without xformers memory optimizations.

## Running the Demo

### Port Configuration (vast.ai)

Check available mapped ports:
```bash
env | grep VAST_TCP_PORT
```

Port 8080 may be occupied by a vast.ai system process. Use an alternative mapped port (e.g., 6006 for TensorBoard):

```bash
GRADIO_LISTEN=1 GRADIO_SERVER_PORT=6006 python demo.py
```

Access from browser at: `http://<vast-public-ip>:<external-mapped-port>`

### API Access

Teammates can call the running Gradio server programmatically:

```python
from gradio_client import Client
client = Client("http://<vast-public-ip>:<external-mapped-port>")
result = client.predict(...)
```

API docs available at: `http://<vast-public-ip>:<external-mapped-port>/?view=api`

## Backup

A clone of the working environment exists:
```bash
conda create -n anytext2_backup --clone anytext2
```

Snapshots were also saved:
```bash
conda list -n anytext2 --export > anytext2_snapshot.txt
pip freeze > anytext2_pip_freeze.txt
conda env config vars list -n anytext2 > anytext2_env_vars.txt
```

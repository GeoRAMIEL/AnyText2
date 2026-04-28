## Backup steps

**1. Capture conda packages**
```bash
conda list -n anytext2 --export > anytext2_snapshot.txt
```

**2. Capture pip packages**
```bash
conda activate anytext2
pip freeze > anytext2_pip_freeze.txt
```

**3. Save the env vars**
```bash
conda env config vars list -n anytext2 > anytext2_env_vars.txt
```

**4. Backup entire conda environment (same-machine only)**
```bash
conda create -n anytext2_backup --clone anytext2
```

## Recovery steps (cross-machine)

The freeze files are reference artifacts, not directly install-able inputs. The conda snapshot has build-hash pins and non-`defaults` channels that don't solve on a fresh box, and the pip freeze has `@ file://` lines from the source machine's conda-built packages plus a date-stamped torch nightly that disappears from the index. Follow the steps below — most of the original freeze versions still apply, with a few documented bumps.

**1. Fresh conda env**
```bash
conda create -n anytext2 python=3.10.6
conda activate anytext2
```

**2. CUDA libs from the nvidia channel**
```bash
conda install -n anytext2 cuda-cudart cuda-libraries cuda-cupti -c nvidia/label/cuda-12.6.0
```

**3. PyTorch nightly cu128** — do NOT pin, the freeze's exact nightly date may be gone:
```bash
pip install --pre torch torchvision \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --force-reinstall
```

**4. Pip packages from the freeze, filtered + `--no-deps`**

Filter out:
- `@ file://` lines (conda-built paths from the source machine)
- `^torch(vision)?==` (handled in step 3)
- `^mkl[-_]` (conda-only MKL shims; PyPI versions don't match)

```bash
grep -vE ' @ file://|^torch(vision)?==|^mkl[-_]' environment/anytext2_pip_freeze.txt \
  > /tmp/anytext2_pip_clean.txt
pip install --no-deps -r /tmp/anytext2_pip_clean.txt
```

`--no-deps` is required: pip's resolver rejects coexistences that the source box ran fine (e.g. gradio metadata says `huggingface-hub>=0.28.1` even though 0.25.1 worked in practice).

**5. Reinstall pure-Python deps that were `@ file://`**

These were conda-installed on the source machine, so they're missing here. Pull them from PyPI:
```bash
pip install pyyaml jinja2 idna six pycparser certifi requests urllib3 \
  filelock networkx sympy charset-normalizer brotlicffi gmpy2 mpmath pysocks
```

**6. Apply documented version bumps**

These are the deltas from the freeze that were necessary on a CUDA-13.2 box (driver newer than the freeze's 13.1):

| Package | Freeze | Recovery | Reason |
|---|---|---|---|
| `huggingface-hub` | 0.25.1 | 0.28.1 | gradio 5.23 requires `>=0.28.1` |
| `diffusers` | 0.25.1 | 0.30.3 | hf-hub 0.26+ removed `cached_download`; diffusers <0.28 still imports it |
| `Pillow` | 9.5.0 | 9.5.0 | Force back to freeze; later versions broke `FreeTypeFont.getsize` (removed in Pillow 10) |
| `fsspec` | (latest) | 2024.2.0 | datasets 2.18 caps at `<=2024.2.0` |
| `numpy` | 1.23.3 | 1.23.5 | scipy 1.15 needs `>=1.23.5`; TF-cpu 2.13 caps at `<=1.24.3` |
| `setuptools` | (latest) | <81 | torch needs `<82`, modelscope needs `pkg_resources` (removed in 82) |

```bash
pip install "huggingface-hub==0.28.1"
pip install --no-deps "diffusers==0.30.3"
pip install --force-reinstall --no-deps "Pillow==9.5.0"
pip install "fsspec==2024.2.0"
pip install "numpy>=1.23.5,<1.24.4"
pip install "setuptools<81"
```

**7. Restore env vars**

`anytext2_env_vars.txt` is a listing, not auto-restorable. Apply with:
```bash
conda env config vars set LD_LIBRARY_PATH=/root/miniconda3/envs/anytext2/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
conda deactivate
conda activate anytext2
```

**8. Re-apply code edits** (xformers disable) per `SETUP_NOTES.md` §6.

**9. Verify with `pip check`** — the only warnings remaining should be:
- `basicsr requires opencv-python` (we use `opencv-python-headless` on headless boxes)
- `tensorflow-cpu requires typing-extensions<4.6.0` (gradio/pydantic need newer; gradio wins)

## Smoke test

```bash
python -c "import torch; print(torch.cuda.is_available()); torch.randn(3).cuda()"
python -c "from diffusers import UNet2DConditionModel; print('ok')"
GRADIO_LISTEN=1 GRADIO_SERVER_PORT=6006 python demo.py
```

---

### The easier option

If you're recovering on the **same machine**, just use the clone:
```bash
conda create -n anytext2 --clone anytext2_backup
```

This avoids all of the cross-machine rebuild dance above.

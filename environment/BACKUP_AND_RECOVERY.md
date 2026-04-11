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

**4. Backup entire conda environment**
```bash
conda create -n anytext2_backup --clone anytext2
```

## Recovery steps:

**1. Create a fresh conda env with the right Python:**
```bash
conda create -n anytext2 python=3.10.6
conda activate anytext2
```

**2. Install conda packages from the snapshot:**
```bash
conda install -n anytext2 --file anytext2_snapshot.txt
```

This will likely fail on some packages (especially the NVIDIA ones from custom channels). For those, install manually:
```bash
conda install -n anytext2 cuda-cudart cuda-libraries cuda-cupti -c nvidia/label/cuda-12.6.0
```

**3. Install pip packages:**
```bash
pip install -r anytext2_pip_freeze.txt
```

**4. Restore env vars:**

The env vars file is just a listing, not auto-restorable. Apply manually based on what's in the file:
```bash
conda env config vars set LD_LIBRARY_PATH=/root/miniconda3/envs/anytext2/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
conda deactivate
conda activate anytext2
```

**5. Re-apply code edits** (xformers disable) per `SETUP_NOTES.md`.

---

### The easier option

If you're recovering on the **same machine**, just use the clone:
```bash
conda create -n anytext2 --clone anytext2_backup
```

The snapshots are most useful when recreating on a **different machine**, where the clone doesn't exist. In that case, `pip freeze` is the most reliable of the three — it captures exact versions of everything pip installed, which is where most of the pain was.

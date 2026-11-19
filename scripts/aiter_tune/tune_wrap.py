"""Run aiter's gemm_moe_tune.py with its hard-coded profile_fmoe.csv redirected
out of the read-only vendored tree (AITER_ROOT_DIR -> FAKE_ROOT)."""
import os, runpy, sys
import aiter.jit.core as core
fake = os.environ["FAKE_ROOT"]
os.makedirs(f"{fake}/aiter/configs", exist_ok=True)
core.AITER_ROOT_DIR = fake
tuner = "/workspace/.vendor/aiter_meta/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py"
sys.argv = [tuner] + sys.argv[1:]
runpy.run_path(tuner, run_name="__main__")

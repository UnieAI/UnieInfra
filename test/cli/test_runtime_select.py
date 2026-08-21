"""Which runtime a checkpoint resolves to, decided from files alone.

``ts serve <model>`` is meant to be the whole command for every model this
engine runs. These pin the mapping, and the ordering between tests that
overlap -- a reranker is also a pooling model, and a diffusers pipeline also
has config.json files inside it.
"""

import json
import subprocess
import sys

import pytest

from tokenspeed.runtime_select import (
    RUNTIMES,
    describe_runtime,
    resolve_serving_runtime,
)


def _checkpoint(root, architectures=None, pooling=False, model_index=False):
    root.mkdir(parents=True, exist_ok=True)
    if architectures is not None:
        (root / "config.json").write_text(json.dumps({"architectures": architectures}))
    if pooling:
        (root / "1_Pooling").mkdir(exist_ok=True)
        (root / "1_Pooling" / "config.json").write_text(
            json.dumps({"pooling_mode_cls_token": True})
        )
    if model_index:
        (root / "model_index.json").write_text(
            json.dumps({"_class_name": "FluxPipeline"})
        )
    return str(root)


def test_generative_llm_is_the_default(tmp_path):
    path = _checkpoint(tmp_path / "c", ["Qwen3ForCausalLM"])
    assert resolve_serving_runtime(path) == "generate"


@pytest.mark.parametrize(
    "arch", ["BertModel", "XLMRobertaModel", "NewModel", "ModernBertModel"]
)
def test_encoder_only_embedding_architectures(tmp_path, arch):
    assert resolve_serving_runtime(_checkpoint(tmp_path / arch, [arch])) == "embed"


def test_a_decoder_is_an_embedder_only_when_the_checkpoint_says_so(tmp_path):
    # Qwen3-Embedding and Qwen3 share an architecture string; 1_Pooling is
    # the only thing separating them, which is why it is tested and not the
    # architecture name.
    generative = _checkpoint(tmp_path / "gen", ["Qwen3ForCausalLM"])
    embedding = _checkpoint(tmp_path / "emb", ["Qwen3ForCausalLM"], pooling=True)
    assert resolve_serving_runtime(generative) == "generate"
    assert resolve_serving_runtime(embedding) == "embed"


def test_cross_encoder_wins_over_pooling(tmp_path):
    # A reranker pools too, but it scores a pair and answers a different
    # route, so the more specific test has to come first.
    path = _checkpoint(
        tmp_path / "rr", ["XLMRobertaForSequenceClassification"], pooling=True
    )
    assert resolve_serving_runtime(path) == "rerank"


def test_whisper_is_asr_not_generate(tmp_path):
    path = _checkpoint(tmp_path / "w", ["WhisperForConditionalGeneration"])
    assert resolve_serving_runtime(path) == "asr"


def test_diffusers_pipeline_wins_over_everything(tmp_path):
    # A pipeline directory also contains component config.json files; the
    # top-level model_index.json is the signal.
    path = _checkpoint(tmp_path / "d", ["Qwen3ForCausalLM"], model_index=True)
    assert resolve_serving_runtime(path) == "diffusion"


@pytest.mark.parametrize("path", ["", "/nonexistent/path/xyz"])
def test_unreadable_metadata_falls_back_rather_than_raising(path):
    # This runs before the loader, which reports a bad checkpoint far better
    # than an argument parser can.
    assert resolve_serving_runtime(path) == "generate"


def test_missing_or_corrupt_config_falls_back(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    assert resolve_serving_runtime(str(root)) == "generate"
    (root / "config.json").write_text("{not json")
    assert resolve_serving_runtime(str(root)) == "generate"


def test_every_runtime_has_a_description():
    for runtime in RUNTIMES:
        assert describe_runtime(runtime) != runtime


def _modules_after(statement: str) -> set[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; {statement}; print(len(sys.modules)); "
            f"print(chr(10).join(sorted(sys.modules)))",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return set(result.stdout.split("\n")[1:])


def test_the_resolver_adds_no_imports_of_its_own():
    """Resolving the runtime must cost nothing beyond ``import tokenspeed``.

    Measured as a delta rather than an absolute, because ``import
    tokenspeed`` already pulls torch: ``tokenspeed/_logging.py`` imports the
    noisy third-party loggers by name in order to filter them, and torch is
    one of them (2.2s, measured 2026-08-21). That is a pre-existing property
    of the package, not of this module -- but it does mean
    ``flat_kvcache_select``'s "before torch is imported" is true of the
    module and not of the import, and the same caveat applies here.

    What this pins is the part that is in this module's control: adding a
    sibling import -- ``tokenspeed.runtime.configs`` is the tempting one --
    would show up as a delta.
    """
    baseline = _modules_after("import tokenspeed")
    with_resolver = _modules_after("import tokenspeed.runtime_select")
    added = {
        name
        for name in with_resolver - baseline
        if not name.startswith("tokenspeed.runtime_select")
    }
    assert added == set(), sorted(added)

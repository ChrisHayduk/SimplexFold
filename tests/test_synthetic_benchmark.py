import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from benchmark_simplexfold import _variant_config, main

from minalphafold.trainer import load_model_config


def test_named_mechanisms_override_disabled_base_without_changing_dimensions():
    base = replace(load_model_config("tiny"), use_simplicial_evoformer=False,
                   simplex_use_faces=False, simplex_use_tetra=False)
    expected = {
        "simplex": (True, True, True, False),
        "faces_only": (True, True, False, False),
        "msa_to_face": (True, True, False, True),
        "no_simplex": (False, False, False, False),
    }
    fields = ("use_simplicial_evoformer", "simplex_use_faces",
              "simplex_use_tetra", "simplex_use_msa_to_face")
    for name, flags in expected.items():
        config = _variant_config(base, name)
        assert tuple(getattr(config, field) for field in fields) == flags
        assert {k: v for k, v in asdict(config).items() if k not in fields} == {
            k: v for k, v in asdict(base).items() if k not in fields
        }


def test_tiny_cpu_benchmark_binds_actual_inputs_and_preserves_existing_output(tmp_path):
    output = tmp_path / "attempt.json"
    argv = ["--device", "cpu", "--length", "8", "--msa-depth", "2",
            "--n-cycles", "1", "--warmup-steps", "0", "--timed-steps", "1",
            "--variants", "simplex", "faces_only", "no_simplex", "--json-out", str(output)]
    rows = main(argv)
    assert json.loads(output.read_text()) == rows
    assert len({row["provenance"]["input_sha256"] for row in rows}) == 1
    assert min(rows[0]["parameters"], rows[1]["parameters"]) > rows[2]["parameters"]
    assert rows[0]["tetras_per_example"] > 0
    assert rows[1]["tetras_per_example"] == rows[2]["tetras_per_example"] == 0
    for row in rows:
        assert row["mean_ms"] > 0
        assert len(row["timings_ms"]) == 1
        assert row["use_simplicial_evoformer"] == row["model_config"]["use_simplicial_evoformer"]
    source = Path(__file__).resolve().parents[1] / "scripts/benchmark_simplexfold.py"
    assert rows[0]["provenance"]["source_sha256"]["scripts/benchmark_simplexfold.py"] == hashlib.sha256(source.read_bytes()).hexdigest()
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(argv)
    assert output.read_bytes() == original


@pytest.mark.parametrize("argv", [["--timed-steps", "0"], ["--warmup-steps", "-1"],
                                 ["--variants", "simplex", "simplex"]])
def test_invalid_benchmark_protocol_rejected_before_measurement(argv):
    with pytest.raises(ValueError):
        main(argv)

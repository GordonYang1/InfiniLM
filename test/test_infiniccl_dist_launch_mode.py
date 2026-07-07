import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "test_infiniccl_dist.py"
_SPEC = importlib.util.spec_from_file_location("test_infiniccl_dist", _SCRIPT)
test_infiniccl_dist = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(test_infiniccl_dist)


def test_standalone_infiniccl_defaults_to_ccl_single_process_without_mpi():
    env = {"INFINILM_USE_INFINICCL": "1"}

    test_infiniccl_dist.validate_launch(tp=2, env=env)


def test_tp_requires_standalone_infiniccl_when_tp_is_greater_than_one():
    with pytest.raises(SystemExit, match="INFINILM_USE_INFINICCL=1"):
        test_infiniccl_dist.validate_launch(tp=2, env={})


def test_explicit_mpi_mode_requires_mpirun_world_size_to_match_tp():
    env = {
        "INFINILM_USE_INFINICCL": "1",
        "INFINILM_INFINICCL_COMM_MODE": "mpi",
        "OMPI_COMM_WORLD_SIZE": "1",
    }

    with pytest.raises(SystemExit, match="mpirun -np 2"):
        test_infiniccl_dist.validate_launch(tp=2, env=env)

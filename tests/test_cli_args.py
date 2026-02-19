"""Tests for the shared CLI argument helper (Issue #2).

Verifies that:
- All standardised flags are present with the correct defaults and types.
- ``validate_benchmark_args`` rejects invalid / incompatible combinations.
- ``build_run_config`` produces a consistent, serialisable dict.
- The flag contract is identical between the suite entry-point and the Q1
  standalone entry-point (parity check).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

# conftest.py adds experiments/ to sys.path so we import directly from
# common.cli_args (no full sage.benchmark namespace needed for lightweight tests).
from common.cli_args import (
    DEFAULT_BACKEND,
    SUPPORTED_BACKENDS,
    add_common_benchmark_args,
    build_run_config,
    validate_benchmark_args,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    return parser.parse_args(argv)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_common_benchmark_args(p)
    return p


# ---------------------------------------------------------------------------
# Presence and defaults
# ---------------------------------------------------------------------------


class TestFlagPresence:
    """All standardised flags must be registered with correct defaults."""

    def test_backend_default(self):
        args = _parse(_make_parser(), [])
        assert args.backend == DEFAULT_BACKEND

    def test_backend_choices(self):
        for backend in SUPPORTED_BACKENDS:
            args = _parse(_make_parser(), ["--backend", backend])
            assert args.backend == backend

    def test_nodes_default(self):
        args = _parse(_make_parser(), [])
        assert args.nodes == 1

    def test_nodes_set(self):
        args = _parse(_make_parser(), ["--nodes", "4"])
        assert args.nodes == 4

    def test_parallelism_default(self):
        args = _parse(_make_parser(), [])
        assert args.parallelism == 2

    def test_parallelism_set(self):
        args = _parse(_make_parser(), ["--parallelism", "8"])
        assert args.parallelism == 8

    def test_repeat_default(self):
        args = _parse(_make_parser(), [])
        assert args.repeat == 1

    def test_repeat_set(self):
        args = _parse(_make_parser(), ["--repeat", "5"])
        assert args.repeat == 5

    def test_seed_default(self):
        args = _parse(_make_parser(), [])
        assert args.seed == 42

    def test_seed_set(self):
        args = _parse(_make_parser(), ["--seed", "0"])
        assert args.seed == 0

    def test_output_dir_default(self):
        args = _parse(_make_parser(), [])
        assert args.output_dir == "results"

    def test_output_dir_set(self):
        args = _parse(_make_parser(), ["--output-dir", "/tmp/my_results"])
        assert args.output_dir == "/tmp/my_results"

    def test_verbose_default_false(self):
        args = _parse(_make_parser(), [])
        assert args.verbose is False

    def test_verbose_flag(self):
        args = _parse(_make_parser(), ["--verbose"])
        assert args.verbose is True

    def test_quick_default_false(self):
        args = _parse(_make_parser(), [])
        assert args.quick is False

    def test_quick_flag(self):
        args = _parse(_make_parser(), ["--quick"])
        assert args.quick is True

    def test_dry_run_default_false(self):
        args = _parse(_make_parser(), [])
        assert args.dry_run is False

    def test_dry_run_flag(self):
        args = _parse(_make_parser(), ["--dry-run"])
        assert args.dry_run is True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """validate_benchmark_args must catch invalid / incompatible combinations."""

    def test_valid_defaults(self):
        args = _parse(_make_parser(), [])
        # Must not raise
        validate_benchmark_args(args)

    def test_repeat_zero_rejected(self):
        args = _parse(_make_parser(), [])
        args.repeat = 0
        with pytest.raises(SystemExit):
            validate_benchmark_args(args)

    def test_repeat_negative_rejected(self):
        args = _parse(_make_parser(), [])
        args.repeat = -1
        with pytest.raises(SystemExit):
            validate_benchmark_args(args)

    def test_parallelism_zero_rejected(self):
        args = _parse(_make_parser(), [])
        args.parallelism = 0
        with pytest.raises(SystemExit):
            validate_benchmark_args(args)

    def test_seed_negative_rejected(self):
        args = _parse(_make_parser(), [])
        args.seed = -1
        with pytest.raises(SystemExit):
            validate_benchmark_args(args)

    def test_nodes_1_with_sage_ok(self):
        args = _parse(_make_parser(), ["--backend", "sage", "--nodes", "1"])
        validate_benchmark_args(args)  # no raise

    def test_nodes_1_with_ray_ok_when_ray_installed(self, monkeypatch):
        monkeypatch.setattr("common.cli_args._module_available", lambda _name: True)
        args = _parse(_make_parser(), ["--backend", "ray", "--nodes", "1"])
        validate_benchmark_args(args)  # no raise

    def test_nodes_multi_with_sage_ok(self):
        args = _parse(_make_parser(), ["--backend", "sage", "--nodes", "4"])
        validate_benchmark_args(args)  # sage supports distributed; no raise

    def test_nodes_multi_with_ray_ok_when_ray_installed(self, monkeypatch):
        monkeypatch.setattr("common.cli_args._module_available", lambda _name: True)
        args = _parse(_make_parser(), ["--backend", "ray", "--nodes", "4"])
        validate_benchmark_args(args)  # ray supports distributed; no raise

    def test_ray_backend_missing_dependency_rejected(self, monkeypatch):
        monkeypatch.setattr("common.cli_args._module_available", lambda _name: False)
        args = _parse(_make_parser(), ["--backend", "ray"])
        with pytest.raises(SystemExit):
            validate_benchmark_args(args)

    def test_repeat_error_message_is_actionable(self, capsys):
        args = _parse(_make_parser(), ["--repeat", "0"])
        with pytest.raises(SystemExit):
            validate_benchmark_args(args)
        err = capsys.readouterr().err
        assert "argument validation failed" in err
        assert "--repeat must be" in err

    def test_ray_missing_dependency_error_is_actionable(self, monkeypatch, capsys):
        monkeypatch.setattr("common.cli_args._module_available", lambda _name: False)
        args = _parse(_make_parser(), ["--backend", "ray"])
        with pytest.raises(SystemExit):
            validate_benchmark_args(args)
        err = capsys.readouterr().err
        assert "Ray backend selected" in err
        assert "python -m pip install -e .[ray-baseline]" in err


# ---------------------------------------------------------------------------
# build_run_config
# ---------------------------------------------------------------------------


class TestBuildRunConfig:
    """build_run_config must produce a stable, serialisable run record."""

    def test_standardised_keys_present(self):
        args = _parse(_make_parser(), [])
        cfg = build_run_config(args)
        for key in ("backend", "nodes", "parallelism", "repeat", "seed", "output_dir"):
            assert key in cfg, f"Missing key: {key}"

    def test_values_match_args(self):
        args = _parse(
            _make_parser(),
            [
                "--backend", "ray",
                "--nodes", "4",
                "--parallelism", "8",
                "--repeat", "3",
                "--seed", "7",
                "--output-dir", "/tmp/out",
            ],
        )
        cfg = build_run_config(args)
        assert cfg["backend"] == "ray"
        assert cfg["nodes"] == 4
        assert cfg["parallelism"] == 8
        assert cfg["repeat"] == 3
        assert cfg["seed"] == 7
        assert cfg["output_dir"] == "/tmp/out"

    def test_extra_kwargs_included(self):
        args = _parse(_make_parser(), [])
        cfg = build_run_config(args, workload="Q1", custom_key="custom_val")
        assert cfg["workload"] == "Q1"
        assert cfg["custom_key"] == "custom_val"

    def test_config_is_json_serialisable(self):
        import json

        args = _parse(_make_parser(), [])
        cfg = build_run_config(args, workload="Q1")
        # Must not raise
        json.dumps(cfg)

    def test_two_backends_same_seed_same_config_keys(self):
        """Equivalent args → equivalent config record structure (parity check)."""
        argv_base = ["--nodes", "1", "--parallelism", "2", "--repeat", "1", "--seed", "42"]
        args_sage = _parse(_make_parser(), argv_base + ["--backend", "sage"])
        args_ray = _parse(_make_parser(), argv_base + ["--backend", "ray"])
        cfg_sage = build_run_config(args_sage, workload="Q1")
        cfg_ray = build_run_config(args_ray, workload="Q1")
        # Same keys
        assert set(cfg_sage.keys()) == set(cfg_ray.keys())
        # Only backend differs
        cfg_sage.pop("backend")
        cfg_ray.pop("backend")
        assert cfg_sage == cfg_ray


# ---------------------------------------------------------------------------
# Parser inclusion guard (backward-compat: no_quick / no_dry_run)
# ---------------------------------------------------------------------------


class TestParserOptions:
    def test_exclude_quick(self):
        p = argparse.ArgumentParser()
        add_common_benchmark_args(p, include_quick=False)
        with pytest.raises(SystemExit):
            p.parse_args(["--quick"])

    def test_exclude_dry_run(self):
        p = argparse.ArgumentParser()
        add_common_benchmark_args(p, include_dry_run=False)
        with pytest.raises(SystemExit):
            p.parse_args(["--dry-run"])


class TestEntryPointParity:
    """Suite + migrated workload should both consume the shared helper."""

    def test_suite_entrypoint_uses_shared_helper(self):
        suite_main = Path(__file__).resolve().parents[1] / "__main__.py"
        content = suite_main.read_text(encoding="utf-8")
        assert "add_common_benchmark_args(parser" in content

    def test_q1_entrypoint_uses_shared_helper(self):
        q1_entry = Path(__file__).resolve().parents[1] / "experiments" / "q1_pipelinechain.py"
        content = q1_entry.read_text(encoding="utf-8")
        assert "add_common_benchmark_args(parser" in content

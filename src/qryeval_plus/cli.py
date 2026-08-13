"""Command-line interface for QryEval_PLUS."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from importlib.resources import files
from pathlib import Path

from qryeval_plus.config import ConfigError, discover_configs, load_config, validate_config
from qryeval_plus.runtime import configure_java_home


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qryeval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one experiment configuration.")
    run.add_argument("config")

    validate = subparsers.add_parser("validate", help="Validate one config or a config directory.")
    validate.add_argument("path")
    validate.add_argument("--no-assets", action="store_true")

    doctor = subparsers.add_parser("doctor", help="Inspect runtime and asset readiness.")
    doctor.add_argument("--config")

    demo = subparsers.add_parser("demo", help="Run a compact local dense-retrieval walkthrough.")
    demo.add_argument("--config", default=None)
    demo.add_argument("--questions", type=int, default=1)
    demo.add_argument("--dry-run", action="store_true")
    return parser


def _validate_command(path: str, check_assets: bool) -> int:
    checked = 0
    failed = 0
    for config_path in discover_configs(path):
        checked += 1
        try:
            parameters = load_config(config_path)
            errors = validate_config(parameters, check_assets=check_assets)
        except Exception as exc:
            errors = [str(exc)]
        if errors:
            failed += 1
            print(f"FAIL {config_path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK   {config_path}")
    print(f"Validated {checked} configuration(s); {failed} failed.")
    return 1 if failed else 0


def _module_status(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _doctor_command(config: str | None) -> int:
    failures = []
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")

    java_home = configure_java_home()
    java_ok = java_home is not None and (java_home / "bin" / "java").is_file()
    print(f"Java: {'OK' if java_ok else 'MISSING'}{f' ({java_home})' if java_home else ''}")
    if not java_ok:
        failures.append("Java runtime")

    for label, module in (
        ("NumPy", "numpy"),
        ("PyTorch", "torch"),
        ("Transformers", "transformers"),
        ("Hugging Face Hub", "huggingface_hub"),
        ("Pyjnius", "jnius"),
        ("FAISS", "faiss"),
    ):
        ok = _module_status(module)
        print(f"{label}: {'OK' if ok else 'MISSING'}")
        if not ok:
            failures.append(label)

    java_dir = files("qryeval_plus").joinpath("vendor", "java")
    jars = list(java_dir.iterdir()) if java_dir.is_dir() else []
    jar_count = len([item for item in jars if item.name.endswith(".jar")])
    print(f"Packaged Java archives: {jar_count}/6")
    if jar_count != 6:
        failures.append("packaged Java archives")

    if config:
        parameters = load_config(config)
        errors = validate_config(parameters, check_assets=True)
        print(f"Configuration: {'OK' if not errors else 'FAILED'} ({Path(config).resolve()})")
        for error in errors:
            print(f"  - {error}")
        failures.extend(errors)

    if failures:
        print("Runtime is not ready: " + ", ".join(failures))
        return 1
    print("Runtime is ready.")
    return 0


def _default_demo_config() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "configs" / "rag" / "systems" / "dense_baseline.json"


def _demo_command(config: str | None, questions: int, dry_run: bool) -> int:
    if questions <= 0:
        raise ConfigError("--questions must be greater than zero.")
    config_path = Path(config).resolve() if config else _default_demo_config()
    parameters = load_config(config_path)
    errors = validate_config(parameters, check_assets=True)
    if errors:
        raise ConfigError("; ".join(errors))

    query_path = Path(parameters["queryFilePath"])
    query_lines = query_path.read_text(encoding="utf-8").splitlines()[:questions]
    print("QryEval_PLUS demo")
    print(f"Configuration: {config_path}")
    print(f"Queries selected: {len(query_lines)} of {len(query_path.read_text(encoding='utf-8').splitlines())}")
    if dry_run:
        print("Dry run complete; no index or model was loaded.")
        return 0

    if questions != len(query_path.read_text(encoding="utf-8").splitlines()):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="qryeval-demo-") as temp_dir:
            query_copy = Path(temp_dir) / "demo.qry"
            query_copy.write_text("\n".join(query_lines) + "\n", encoding="utf-8")
            parameters["queryFilePath"] = str(query_copy)
            from qryeval_plus.pipeline import run_pipeline
            batch = run_pipeline(parameters)
    else:
        from qryeval_plus.pipeline import run_pipeline
        batch = run_pipeline(parameters)
    for qid, qinfo in batch.items():
        print(f"{qid}: {qinfo.get('answer', '')}")
    return 0


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "run":
            from qryeval_plus.pipeline import run_config
            run_config(args.config)
            code = 0
        elif args.command == "validate":
            code = _validate_command(args.path, not args.no_assets)
        elif args.command == "doctor":
            code = _doctor_command(args.config)
        else:
            code = _demo_command(args.config, args.questions, args.dry_run)
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)

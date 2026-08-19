"""Command-line interface for QryEval_PLUS."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import platform
import sys
from importlib.resources import files
from pathlib import Path

from qryeval_plus.config import (
    ConfigError,
    discover_configs,
    load_config,
    ordered_tasks,
    validate_config,
)
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

    benchmark = subparsers.add_parser(
        "benchmark", help="Run a reproducible fixed-vs-agent benchmark."
    )
    benchmark.add_argument("manifest")
    benchmark.add_argument("--limit", type=int, default=None)
    benchmark.add_argument("--split", choices=("dev", "test", "all"))
    benchmark.add_argument("--resume", action="store_true")
    benchmark.add_argument("--dry-run-budget", action="store_true")
    benchmark.add_argument("--publish-to", help="Publish scrubbed lightweight evidence after a completed run.")

    corpus = subparsers.add_parser("corpus", help="Inspect mounted corpus assets.")
    corpus_sub = corpus.add_subparsers(dest="corpus_command", required=True)
    inspect = corpus_sub.add_parser("inspect", help="Write Lucene/FAISS corpus metadata.")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--output")

    dataset = subparsers.add_parser("dataset", help="Prepare or validate dataset protocols.")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    prepare = dataset_sub.add_parser("prepare", help="Build the frozen TriviaQA 40/278 split.")
    prepare.add_argument("--gold", required=True)
    prepare.add_argument("--dev-queries", required=True)
    prepare.add_argument("--output-dir", required=True)
    check = dataset_sub.add_parser("validate", help="Validate a frozen TriviaQA split manifest.")
    check.add_argument("--manifest", required=True)
    check.add_argument("--gold", required=True)
    check.add_argument("--qrels", required=True)

    serve = subparsers.add_parser("serve", help="Run the local QryEval HTTP service.")
    serve.add_argument("--config", required=True)
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)

    loadtest = subparsers.add_parser("loadtest", help="Run a bounded HTTP load test.")
    loadtest.add_argument("--url", default="http://127.0.0.1:8000")
    loadtest.add_argument("--questions", required=True)
    loadtest.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    loadtest.add_argument("--requests", type=int, default=200)
    loadtest.add_argument("--policy", choices=("fixed_bm25", "adaptive_rewrite"), default="adaptive_rewrite")
    loadtest.add_argument("--token-env")
    loadtest.add_argument("--output")
    loadtest.add_argument("--cache-bust", action="store_true", help="Append a unique suffix to measure uncached infrastructure throughput.")
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
        if not errors:
            from qryeval_plus.llm import provider_summary

            agentic = any(
                role == "agent" and str(task.get("type", "")).lower() == "agentic_rag"
                for _, role, task in ordered_tasks(parameters)
            )
            if agentic:
                python_ok = sys.version_info[:2] == (3, 11)
                print(f"Agent Python 3.11: {'OK' if python_ok else 'MISMATCH'}")
                if not python_ok:
                    failures.append("Python 3.11 Agent runtime")
                for label, module in (("Pydantic", "pydantic"), ("LangGraph", "langgraph")):
                    ok = _module_status(module)
                    print(f"{label}: {'OK' if ok else 'MISSING'}")
                    if not ok:
                        failures.append(label)

            for task_name, role, task in ordered_tasks(parameters):
                if role != "agent" or str(task.get("type", "")).lower() not in {
                    "rag", "agentic_rag"
                }:
                    continue
                summary = provider_summary(task)
                print(
                    "LLM: {} / {} ({})".format(
                        summary["provider"], summary["model"], summary["base_url"]
                    )
                )
                key_name = summary["api_key_env"]
                key_set = bool(os.environ.get(key_name, "").strip())
                print(f"LLM credential {key_name}: {'SET' if key_set else 'MISSING'}")
                if summary["api_key_required"] and not key_set:
                    failures.append(f"environment variable {key_name}")

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
        elif args.command == "demo":
            code = _demo_command(args.config, args.questions, args.dry_run)
        elif args.command == "benchmark":
            from qryeval_plus.benchmark import estimate_budget, publish_artifacts, run_benchmark
            limit = args.limit if (args.limit is not None or args.split) else 5
            if args.dry_run_budget:
                print(json.dumps(estimate_budget(args.manifest, split=args.split, limit=limit), indent=2))
            else:
                output = run_benchmark(
                    args.manifest, limit=limit, split=args.split, resume=args.resume
                )
                print("Benchmark artifacts: {}".format(output))
                if args.publish_to:
                    print("Published evidence: {}".format(publish_artifacts(output, args.publish_to)))
            code = 0
        elif args.command == "corpus":
            from qryeval_plus.protocol import inspect_corpus
            payload = inspect_corpus(args.config)
            rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            if args.output:
                Path(args.output).expanduser().resolve().write_text(rendered, encoding="utf-8")
                print("Corpus manifest: {}".format(Path(args.output).expanduser().resolve()))
            else:
                print(rendered, end="")
            code = 0
        elif args.command == "dataset":
            from qryeval_plus.protocol import validate_split_manifest, write_split_files
            if args.dataset_command == "prepare":
                payload = write_split_files(args.gold, args.dev_queries, args.output_dir)
            else:
                payload = validate_split_manifest(args.manifest, args.gold, args.qrels)
            print(json.dumps(payload, indent=2, sort_keys=True))
            code = 0
        elif args.command == "serve":
            if sys.version_info < (3, 11):
                raise RuntimeError("The service requires Python 3.11 or newer.")
            import uvicorn
            from qryeval_plus.service import create_app
            settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
            host = args.host or settings.get("host", "127.0.0.1")
            port = args.port or int(settings.get("port", 8000))
            uvicorn.run(create_app(args.config), host=host, port=port, workers=1)
            code = 0
        elif args.command == "loadtest":
            from qryeval_plus.loadtest import read_questions, run_load_test
            questions = read_questions(args.questions)
            token = os.environ.get(args.token_env) if args.token_env else None
            reports = [
                asyncio.run(run_load_test(
                    url=args.url, questions=questions, concurrency=value,
                    requests=args.requests, policy=args.policy, token=token,
                    cache_bust=args.cache_bust,
                    cache_bust_label="c{}".format(value),
                ))
                for value in args.concurrency
            ]
            rendered = json.dumps({"runs": reports}, indent=2, sort_keys=True) + "\n"
            if args.output:
                Path(args.output).expanduser().resolve().write_text(rendered, encoding="utf-8")
                print("Load-test report: {}".format(Path(args.output).expanduser().resolve()))
            else:
                print(rendered, end="")
            code = 0
        else:
            raise ConfigError("Unsupported command: {}".format(args.command))
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)

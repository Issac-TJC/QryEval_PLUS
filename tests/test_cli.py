import os
import subprocess
import sys

from conftest import PROJECT_ROOT


def run_cli(*args, cwd=None, extra_env=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", "qryeval_plus", *args],
        cwd=cwd or PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_validate_and_demo_work_from_another_directory(tmp_path):
    config = PROJECT_ROOT / "configs" / "rag" / "systems" / "dense_baseline.json"
    validation = run_cli("validate", str(config), cwd=tmp_path)
    assert validation.returncode == 0, validation.stdout + validation.stderr

    demo = run_cli("demo", "--config", str(config), "--dry-run", cwd=tmp_path)
    assert demo.returncode == 0, demo.stdout + demo.stderr
    assert "Dry run complete" in demo.stdout


def test_invalid_run_and_missing_config_return_failure(tmp_path):
    missing = tmp_path / "missing.json"
    result = run_cli("run", str(missing), cwd=tmp_path)
    assert result.returncode == 1
    assert "does not exist" in result.stderr


def test_run_command_succeeds_from_another_directory(tmp_path):
    config = PROJECT_ROOT / "configs" / "examples" / "retrieval_smoke.json"
    result = run_cli("run", str(config), cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (PROJECT_ROOT / "outputs" / "examples" / "retrieval_smoke.run").is_file()


def test_doctor_reports_ready_runtime():
    config = PROJECT_ROOT / "configs" / "rag" / "systems" / "dense_baseline.json"
    result = run_cli(
        "doctor",
        "--config",
        str(config),
        extra_env={"DEEPSEEK_API_KEY": "test-only-key"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Runtime is ready" in result.stdout
    assert "LLM: deepseek / deepseek-v4-flash" in result.stdout


def test_doctor_reports_missing_llm_credential():
    config = PROJECT_ROOT / "configs" / "rag" / "systems" / "dense_baseline.json"
    result = run_cli(
        "doctor",
        "--config",
        str(config),
        extra_env={"DEEPSEEK_API_KEY": ""},
    )
    assert result.returncode == 1
    assert "LLM credential DEEPSEEK_API_KEY: MISSING" in result.stdout

"""Runtime discovery helpers."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def configure_java_home() -> Path | None:
    """Discover a JVM before Pyjnius is imported."""
    configured = os.environ.get("JAVA_HOME")
    if configured:
        path = Path(configured).expanduser()
        if (path / "bin" / "java").is_file():
            return path.resolve()

    conda_jvm = Path(sys.prefix) / "lib" / "jvm"
    if (conda_jvm / "bin" / "java").is_file():
        os.environ["JAVA_HOME"] = str(conda_jvm)
        return conda_jvm.resolve()

    java = shutil.which("java")
    if java:
        java_path = Path(java).resolve()
        home = java_path.parent.parent
        os.environ.setdefault("JAVA_HOME", str(home))
        return home
    return None


configure_java_home()

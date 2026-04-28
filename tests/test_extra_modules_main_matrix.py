import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_extra_modules_main_matrix_cli_generates_report(tmp_path: Path):
    report_json = tmp_path / "extra_modules_main_matrix.json"
    report_md = tmp_path / "extra_modules_main_matrix.md"
    command = [
        sys.executable,
        "tests/extra_modules/run_main_matrix.py",
        "--max-files",
        "2",
        "--workers",
        "1",
        "--timeout",
        "90",
        "--report-json",
        str(report_json),
        "--report-md",
        str(report_md),
        "--no-require-cuda",
        "--no-require-triton",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr[-800:]
    assert report_json.exists()
    assert report_md.exists()

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert "config" in payload
    assert "summary" in payload
    assert "records" in payload
    assert isinstance(payload["records"], list)
    assert payload["summary"]["total"] == len(payload["records"])

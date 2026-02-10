"""PBS job generation and submission."""

import subprocess
import sys
import tempfile
from pathlib import Path

from jinja2 import Environment, PackageLoader

from .config import AegisConfig


def _get_template_env() -> Environment:
    return Environment(
        loader=PackageLoader("aegis", "templates"),
        keep_trailing_newline=True,
    )


def generate_pbs_script(config: AegisConfig) -> str:
    """Render the PBS batch script from the Jinja2 template."""
    env = _get_template_env()
    template = env.get_template("pbs_job.sh.j2")
    return template.render(config=config)


def submit_job(script: str) -> str:
    """Write the PBS script to a temp file and submit via qsub. Returns the job ID."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pbs", prefix="aegis_", delete=False
    ) as f:
        f.write(script)
        script_path = f.name

    print(f"Submitting PBS script: {script_path}", file=sys.stderr)
    result = subprocess.run(
        ["qsub", script_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"qsub failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    job_id = result.stdout.strip()
    print(f"Submitted job: {job_id}", file=sys.stderr)
    return job_id

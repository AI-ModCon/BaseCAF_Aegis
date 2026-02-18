"""PBS job generation and submission."""

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from jinja2 import Environment, PackageLoader

from .config import AegisConfig, config_to_yaml


def _get_template_env() -> Environment:
    return Environment(
        loader=PackageLoader("aegis", "templates"),
        keep_trailing_newline=True,
    )


def generate_pbs_script(config: AegisConfig) -> str:
    """Render the PBS batch script from the Jinja2 template."""
    env = _get_template_env()
    template = env.get_template("pbs_job.sh.j2")
    config_yaml = config_to_yaml(config)
    return template.render(config=config, config_yaml=config_yaml)


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


class SSHConnection:
    """Manages an SSH ControlMaster session for OTP-based hosts."""

    def __init__(self, remote: str):
        self.remote = remote
        self.socket_path = f"/tmp/aegis-ssh-{uuid4().hex[:8]}"

    def connect(self) -> None:
        """Open a ControlMaster session. The user will be prompted for OTP."""
        print(f"Opening SSH connection to {self.remote} ...", file=sys.stderr)
        result = subprocess.run(
            [
                "ssh", "-M",
                "-S", self.socket_path,
                "-o", "ControlPersist=yes",
                self.remote, "true",
            ],
        )
        if result.returncode != 0:
            print("SSH connection failed.", file=sys.stderr)
            sys.exit(1)
        print("SSH connection established.", file=sys.stderr)

    def run(self, command: str) -> subprocess.CompletedProcess:
        """Run a command on the remote host over the existing connection."""
        return subprocess.run(
            [
                "ssh",
                "-S", self.socket_path,
                self.remote,
                command,
            ],
            capture_output=True,
            text=True,
        )

    def scp_to(self, local_path: str, remote_path: str) -> None:
        """Copy a local file to the remote host."""
        result = subprocess.run(
            [
                "scp",
                "-o", f"ControlPath={self.socket_path}",
                local_path,
                f"{self.remote}:{remote_path}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"scp failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

    def close(self) -> None:
        """Tear down the ControlMaster session."""
        subprocess.run(
            [
                "ssh",
                "-S", self.socket_path,
                "-O", "exit",
                self.remote,
            ],
            capture_output=True,
            text=True,
        )


def submit_job_remote(script: str, ssh: SSHConnection) -> str:
    """Submit a PBS job via an SSH connection. Returns the job ID."""
    remote_script = f"~/.{uuid4().hex[:8]}.aegis.pbs"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pbs", prefix="aegis_", delete=False
    ) as f:
        f.write(script)
        local_path = f.name

    print(f"Copying PBS script to {ssh.remote}:{remote_script}", file=sys.stderr)
    ssh.scp_to(local_path, remote_script)

    print("Submitting PBS job via SSH ...", file=sys.stderr)
    result = ssh.run(f"qsub {remote_script} && rm -f {remote_script}")
    if result.returncode != 0:
        # Clean up remote file on failure too
        ssh.run(f"rm -f {remote_script}")
        print(f"Remote qsub failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    job_id = result.stdout.strip()
    print(f"Submitted job: {job_id}", file=sys.stderr)
    return job_id


def _check_job_running(job_id: str, ssh: SSHConnection | None = None) -> bool:
    """Return True if qstat reports the job exists (queued or running)."""
    cmd = f"qstat {job_id}"
    if ssh:
        result = ssh.run(cmd)
    else:
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
    return result.returncode == 0


def _read_endpoints_file(
    endpoints_file: str, ssh: SSHConnection | None = None
) -> list[str] | None:
    """Return endpoint lines if the file exists and is non-empty, else None."""
    if ssh:
        result = ssh.run(f"test -s {endpoints_file} && cat {endpoints_file}")
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()
        return None
    else:
        path = Path(endpoints_file)
        if path.is_file() and path.stat().st_size > 0:
            return path.read_text().strip().splitlines()
        return None


def wait_for_endpoints(
    endpoints_file: str,
    job_id: str,
    poll_interval: int = 15,
    ssh: SSHConnection | None = None,
) -> list[str]:
    """Poll until the endpoints file appears or the job dies.

    Returns the list of endpoint strings on success.
    Exits with code 1 if the job terminates before endpoints are written.
    """
    print(
        f"Waiting for endpoints file: {endpoints_file} (job {job_id})",
        file=sys.stderr,
    )

    while True:
        # Check if endpoints file is ready
        endpoints = _read_endpoints_file(endpoints_file, ssh)
        if endpoints:
            print("Endpoints:", file=sys.stderr)
            for ep in endpoints:
                print(ep)
            return endpoints

        # Check if job is still alive
        if not _check_job_running(job_id, ssh):
            print(
                f"Error: Job {job_id} is no longer running and endpoints file "
                f"was not found.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"  Job {job_id} is running, waiting ...", file=sys.stderr)
        time.sleep(poll_interval)

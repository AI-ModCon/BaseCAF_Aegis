"""PBS job generation and submission."""

import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse
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


def submit_job(script: str, hf_token: str | None = None) -> str:
    """Write the PBS script to a temp file and submit via qsub. Returns the job ID."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pbs", prefix="aegis_", delete=False
    ) as f:
        f.write(script)
        script_path = f.name

    qsub_cmd = ["qsub"]
    if hf_token:
        qsub_cmd += ["-v", "HF_TOKEN"]
    qsub_cmd.append(script_path)

    print(f"Submitting PBS script: {script_path}", file=sys.stderr)
    result = subprocess.run(
        qsub_cmd,
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
        """Run a command on the remote host over the existing connection.

        The command is wrapped in a login shell so that the user's profile
        (module loads, PATH additions, etc.) is sourced.
        """
        return subprocess.run(
            [
                "ssh",
                "-S", self.socket_path,
                self.remote,
                f"bash -l -c {shlex.quote(command)}",
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

    def scp_from(self, remote_path: str, local_path: str) -> None:
        """Copy a file from the remote host to the local machine."""
        result = subprocess.run(
            [
                "scp",
                "-o", f"ControlPath={self.socket_path}",
                f"{self.remote}:{remote_path}",
                local_path,
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


def submit_job_remote(script: str, ssh: SSHConnection, hf_token: str | None = None) -> str:
    """Submit a PBS job via an SSH connection. Returns the job ID."""
    remote_script = f"~/.{uuid4().hex[:8]}.aegis.pbs"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pbs", prefix="aegis_", delete=False
    ) as f:
        f.write(script)
        local_path = f.name

    print(f"Copying PBS script to {ssh.remote}:{remote_script}", file=sys.stderr)
    ssh.scp_to(local_path, remote_script)

    qsub_cmd = "qsub"
    if hf_token:
        qsub_cmd += f" -v HF_TOKEN={shlex.quote(hf_token)}"

    print("Submitting PBS job via SSH ...", file=sys.stderr)
    result = ssh.run(f"{qsub_cmd} {remote_script} && rm -f {remote_script}")
    if result.returncode != 0:
        # Clean up remote file on failure too
        ssh.run(f"rm -f {remote_script}")
        print(f"Remote qsub failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    job_id = result.stdout.strip()
    print(f"Submitted job: {job_id}", file=sys.stderr)
    return job_id


def _get_job_state(job_id: str, ssh: SSHConnection | None = None) -> str | None:
    """Return the PBS job state character (Q, R, E, H, etc.) or None if the
    job is no longer tracked by the scheduler."""
    cmd = f"qstat -f {job_id}"
    if ssh:
        result = ssh.run(cmd)
    else:
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("job_state"):
            # Format: "job_state = R"
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None


_JOB_STATE_LABELS = {
    "Q": "queued",
    "R": "running",
    "H": "held",
    "E": "exiting",
    "T": "moving",
    "W": "waiting",
    "S": "suspended",
}


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


def _read_registry_url(
    endpoints_file: str, ssh: SSHConnection | None = None
) -> str | None:
    """Read the registry URL sidecar file written by the launcher."""
    registry_file = str(Path(endpoints_file).parent / "aegis_registry_url.txt")
    if ssh:
        result = ssh.run(f"test -s {registry_file} && cat {registry_file}")
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    else:
        path = Path(registry_file)
        if path.is_file() and path.stat().st_size > 0:
            return path.read_text().strip()
        return None


def wait_for_endpoints(
    endpoints_file: str,
    job_id: str,
    poll_interval: int = 15,
    ssh: SSHConnection | None = None,
) -> list[str]:
    """Poll until the endpoints file appears or the job dies.

    Returns the list of endpoint strings on success.  When *ssh* is provided
    the remote file is also copied to a local file with the same basename in
    the current working directory.
    Exits with code 1 if the job terminates before endpoints are written.
    """
    print(
        f"Waiting for endpoints file: {endpoints_file} (job {job_id})",
        file=sys.stderr,
    )

    last_state = None
    start = time.monotonic()

    while True:
        # Check if endpoints file is ready
        endpoints = _read_endpoints_file(endpoints_file, ssh)
        if endpoints:
            # Copy the remote file locally so users have it on disk
            if ssh:
                local_path = Path(endpoints_file).name
                ssh.scp_from(endpoints_file, local_path)
                print(
                    f"Endpoints file copied to ./{local_path}",
                    file=sys.stderr,
                )

            elapsed = int(time.monotonic() - start)
            minutes, seconds = divmod(elapsed, 60)
            print(
                f"\nLaunch complete ({minutes}m{seconds:02d}s). "
                f"{len(endpoints)} endpoint(s) ready.",
                file=sys.stderr,
            )

            # Print registry URL and usage hint
            registry_url = _read_registry_url(endpoints_file, ssh)
            if registry_url:
                if ssh:
                    registry_local = Path(endpoints_file).parent / "aegis_registry_url.txt"
                    ssh.scp_from(
                        str(Path(endpoints_file).parent / "aegis_registry_url.txt"),
                        str(registry_local.name),
                    )
                print(f"\nRegistry: {registry_url}", file=sys.stderr)
                parsed = urlparse(registry_url)
                print(
                    f"  aegis registry list-healthy "
                    f"--registry-host {parsed.hostname} "
                    f"--registry-port {parsed.port}",
                    file=sys.stderr,
                )

            return endpoints

        # Check if job is still alive
        state = _get_job_state(job_id, ssh)
        if state is None:
            print(
                f"\nError: Job {job_id} is no longer tracked by the scheduler "
                f"and endpoints file was not found.",
                file=sys.stderr,
            )
            sys.exit(1)

        label = _JOB_STATE_LABELS.get(state, state)
        elapsed = int(time.monotonic() - start)
        minutes, seconds = divmod(elapsed, 60)

        if state != last_state:
            # Print new state on its own line
            print(
                f"\n  Job is {label} ({minutes}m{seconds:02d}s elapsed)",
                file=sys.stderr,
                end="",
            )
            last_state = state
        else:
            # Same state — just print a dot to show progress
            print(".", file=sys.stderr, end="", flush=True)

        time.sleep(poll_interval)


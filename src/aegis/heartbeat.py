"""Per-instance heartbeat monitor for vLLM health tracking."""

import sys
import time
import urllib.request
import urllib.error

from .registry import ServiceRegistry, ServiceStatus


def run_heartbeat(
    service_id: str,
    host: str,
    port: int,
    redis_host: str,
    redis_port: int,
    interval: int = 30,
) -> None:
    """Poll a vLLM instance's /health endpoint and update the registry.

    This function runs an infinite loop and is meant to be invoked as a
    standalone subprocess via ``python -m aegis.heartbeat``.
    """
    registry = ServiceRegistry(redis_host=redis_host, redis_port=redis_port)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://{host}:{port}/health"

    last_status: ServiceStatus | None = None

    while True:
        try:
            with opener.open(url, timeout=10) as resp:
                status = ServiceStatus.HEALTHY if resp.status == 200 else ServiceStatus.UNHEALTHY
        except (urllib.error.URLError, OSError):
            status = ServiceStatus.UNHEALTHY

        if status != last_status:
            print(
                f"[heartbeat] {service_id}: {last_status.value if last_status else 'init'}"
                f" -> {status.value}",
                file=sys.stderr,
            )
            last_status = status

        registry.update_health(service_id, status)
        time.sleep(interval)


def run_heartbeat_all(
    endpoints: list[tuple[str, str, int]],
    redis_host: str,
    redis_port: int,
    interval: int = 30,
) -> None:
    """Monitor multiple vLLM instances from a single process.

    *endpoints* is a list of ``(service_id, host, port)`` tuples.  Each
    cycle checks every endpoint's ``/health`` route, updates the Redis
    registry, then sleeps for *interval* seconds.
    """
    registry = ServiceRegistry(redis_host=redis_host, redis_port=redis_port)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    last_statuses: dict[str, ServiceStatus] = {}

    while True:
        for service_id, host, port in endpoints:
            url = f"http://{host}:{port}/health"
            try:
                with opener.open(url, timeout=10) as resp:
                    status = ServiceStatus.HEALTHY if resp.status == 200 else ServiceStatus.UNHEALTHY
            except (urllib.error.URLError, OSError):
                status = ServiceStatus.UNHEALTHY

            last_status = last_statuses.get(service_id)
            if status != last_status:
                print(
                    f"[heartbeat] {service_id}: {last_status.value if last_status else 'init'}"
                    f" -> {status.value}",
                    file=sys.stderr,
                )
                last_statuses[service_id] = status

            registry.update_health(service_id, status)

        time.sleep(interval)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--all":
        # Multi-instance mode:
        #   python -m aegis.heartbeat --all REDIS_HOST REDIS_PORT SVC:HOST:PORT [...]
        if len(sys.argv) < 5:
            print(
                "Usage: python -m aegis.heartbeat --all REDIS_HOST REDIS_PORT"
                " SVC_ID:HOST:PORT [SVC_ID:HOST:PORT ...]",
                file=sys.stderr,
            )
            sys.exit(1)

        _redis_host = sys.argv[2]
        _redis_port = int(sys.argv[3])
        _endpoints: list[tuple[str, str, int]] = []
        for arg in sys.argv[4:]:
            svc_id, host, port_str = arg.split(":")
            _endpoints.append((svc_id, host, int(port_str)))

        run_heartbeat_all(_endpoints, _redis_host, _redis_port)
    else:
        # Single-instance mode (backwards compatible):
        #   python -m aegis.heartbeat SERVICE_ID HOST PORT REDIS_HOST REDIS_PORT [INTERVAL]
        if len(sys.argv) < 6:
            print(
                "Usage: python -m aegis.heartbeat SERVICE_ID HOST PORT REDIS_HOST REDIS_PORT [INTERVAL]",
                file=sys.stderr,
            )
            sys.exit(1)

        _service_id = sys.argv[1]
        _host = sys.argv[2]
        _port = int(sys.argv[3])
        _redis_host = sys.argv[4]
        _redis_port = int(sys.argv[5])
        _interval = int(sys.argv[6]) if len(sys.argv) > 6 else 30

        run_heartbeat(_service_id, _host, _port, _redis_host, _redis_port, _interval)

"""Per-instance heartbeat monitor with embedded in-process registry."""

import sys
import time
import urllib.request
import urllib.error

from .registry import InMemoryRegistry, ServiceInfo, ServiceStatus, start_registry_server


def run_heartbeat_all(
    endpoints: list[tuple[str, str, int]],
    registry: InMemoryRegistry,
    interval: int = 30,
) -> None:
    """Monitor multiple vLLM instances from a single process.

    *endpoints* is a list of ``(service_id, host, port)`` tuples.  Each
    cycle checks every endpoint's ``/health`` route, updates the in-memory
    registry, then sleeps for *interval* seconds.
    """
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
    # Usage:
    #   python -m aegis.heartbeat --registry-port PORT SVC_ID:HOST:PORT [...]
    if len(sys.argv) < 4 or sys.argv[1] != "--registry-port":
        print(
            "Usage: python -m aegis.heartbeat --registry-port PORT"
            " SVC_ID:HOST:PORT [SVC_ID:HOST:PORT ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    registry_port = int(sys.argv[2])
    _endpoints: list[tuple[str, str, int]] = []
    _endpoint_models: dict[tuple[str, int], str] = {}
    for arg in sys.argv[3:]:
        parts = arg.split(":", 3)
        svc_id, host, port_str = parts[0], parts[1], parts[2]
        model = parts[3] if len(parts) > 3 else ""
        _endpoints.append((svc_id, host, int(port_str)))
        _endpoint_models[(host, int(port_str))] = model

    # Create in-memory registry and register all instances as STARTING
    _registry = InMemoryRegistry()
    for svc_id, host, port in _endpoints:
        info = ServiceInfo(
            service_id=svc_id,
            host=host,
            port=port,
            service_type="vllm",
            status=ServiceStatus.STARTING.value,
            metadata={"model": _endpoint_models[(host, port)]},
        )
        _registry.register_service(info)

    # Start the HTTP query server
    start_registry_server(_registry, host="0.0.0.0", port=registry_port)
    print(f"Registry server listening on 0.0.0.0:{registry_port}", file=sys.stderr)

    # Run the heartbeat loop (blocks forever)
    run_heartbeat_all(_endpoints, _registry)

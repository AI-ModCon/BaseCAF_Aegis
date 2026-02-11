#!/bin/bash
# Stop Redis server

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# REDIS_STABLE="${SCRIPT_DIR}/redis-stable"
# This should be set in env.sh
PID_FILE="${SCRIPT_DIR}/redis-server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Redis PID file not found at: $PID_FILE"
    exit 1
fi

PID=$(cat "$PID_FILE")
echo "$(date) Stopping Redis server (PID: $PID)"

# Locate redis-cli: prefer REDIS_STABLE if set, otherwise fall back to PATH
if [ -n "${REDIS_STABLE}" ] && [ -x "${REDIS_STABLE}/src/redis-cli" ]; then
    REDIS_CLI="${REDIS_STABLE}/src/redis-cli"
else
    REDIS_CLI="$(command -v redis-cli 2>/dev/null)"
fi

# Try graceful shutdown first
${REDIS_CLI:-redis-cli} shutdown 2>/dev/null || kill -TERM $PID

# Wait for shutdown
for i in {1..10}; do
    if ! kill -0 $PID 2>/dev/null; then
        echo "$(date) Redis server stopped successfully"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
if kill -0 $PID 2>/dev/null; then
    echo "$(date) Force killing Redis server"
    kill -9 $PID
    rm -f "$PID_FILE"
fi

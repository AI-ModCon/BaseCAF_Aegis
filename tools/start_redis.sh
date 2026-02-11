#!/bin/bash
# Start Redis server bound to a specific network interface
# Usage: ./start_redis.sh [bind_address] [port]

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# This should be set in env.sh
# REDIS_STABLE="${SCRIPT_DIR}/redis-stable"

# Get bind address (default to compute node network)
# On Aurora, you'll want the HSN (high-speed network) address
BIND_ADDRESS=${1:-$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | head -n 1)}
REDIS_PORT=${2:-6379}

# If BIND_ADDRESS is still empty, fall back to primary IP
if [ -z "$BIND_ADDRESS" ]; then
    BIND_ADDRESS=$(hostname -i | awk '{print $1}')
fi

echo "$(date) Starting Redis server"
echo "$(date)   Bind address: ${BIND_ADDRESS}"
echo "$(date)   Port: ${REDIS_PORT}"

# Locate redis-server: prefer REDIS_STABLE if set, otherwise fall back to PATH
if [ -n "${REDIS_STABLE}" ] && [ -x "${REDIS_STABLE}/src/redis-server" ]; then
    REDIS_SERVER="${REDIS_STABLE}/src/redis-server"
    REDIS_CONF="${REDIS_STABLE}/redis.conf"
else
    REDIS_SERVER="$(command -v redis-server 2>/dev/null)"
    REDIS_CONF=""
    if [ -z "$REDIS_SERVER" ]; then
        echo "$(date) ERROR: redis-server not found. Set REDIS_STABLE or add redis-server to PATH."
        exit 1
    fi
fi

# Start Redis with command-line overrides
${REDIS_SERVER} ${REDIS_CONF} \
    --bind ${BIND_ADDRESS} \
    --port ${REDIS_PORT} \
    --protected-mode no \
    --daemonize yes \
    --logfile ${SCRIPT_DIR}/redis-server.log \
    --pidfile ${SCRIPT_DIR}/redis-server.pid

# Wait a moment and check if it started
sleep 2

if [ -f "${SCRIPT_DIR}/redis-server.pid" ]; then
    PID=$(cat ${SCRIPT_DIR}/redis-server.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "$(date) Redis server started successfully (PID: $PID)"
        echo "$(date) Connect with: redis-cli -h ${BIND_ADDRESS} -p ${REDIS_PORT}"
        echo "$(date) Logs at: ${SCRIPT_DIR}/redis-server.log"
        exit 0
    else
        echo "$(date) ERROR: Redis server failed to start"
        cat ${SCRIPT_DIR}/redis-server.log
        exit 1
    fi
else
    echo "$(date) ERROR: Redis server PID file not created"
    exit 1
fi

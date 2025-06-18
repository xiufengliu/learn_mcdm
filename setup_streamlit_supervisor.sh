#!/bin/bash

set -e  # Exit immediately on any error

# === Configurable Variables ===
APP_NAME="learn_mcdm"
APP_DIR="/work/Projects/learn_mcdm"
APP_ENTRY="app.py"
LOG_DIR="${APP_DIR}/logs"
SUPERVISOR_CONF="/etc/supervisor/conf.d/${APP_NAME}.conf"
STREAMLIT_BIN="/work/miniconda3/bin/streamlit"
RUN_USER="ucloud"
PORT=80

# === Step 0: Ensure Supervisor is Installed ===
echo "Checking if supervisor is installed..."
if ! command -v supervisord &> /dev/null; then
    echo "Supervisor not found. Installing..."
    sudo apt update
    sudo apt install -y supervisor
else
    echo "Supervisor is already installed."
fi

# === Step 1: Create logs directory ===
echo "Creating log directory at $LOG_DIR..."
sudo mkdir -p "$LOG_DIR"
sudo chown -R "$RUN_USER:$RUN_USER" "$LOG_DIR"

# === Step 2: Create Supervisor Config ===
echo "Writing Supervisor config to $SUPERVISOR_CONF..."
sudo bash -c "cat > $SUPERVISOR_CONF" <<EOF
[program:${APP_NAME}]
directory=${APP_DIR}
command=${STREAMLIT_BIN} run ${APP_ENTRY} --server.address=0.0.0.0 --server.port=${PORT}
autostart=true
autorestart=true
startsecs=10
stdout_logfile=${LOG_DIR}/${APP_NAME}.out.log
stderr_logfile=${LOG_DIR}/${APP_NAME}.err.log
user=${RUN_USER}
environment=PATH="/work/miniconda3/bin",PYTHONUNBUFFERED=1
EOF

# === Step 3: Start supervisord if not already running ===
if ! pgrep -x "supervisord" > /dev/null; then
    echo "Starting supervisord..."
    sudo supervisord -c /etc/supervisor/supervisord.conf
else
    echo "supervisord is already running."
fi

# === Step 4: Reload and Start Streamlit App ===
echo "Reloading and updating supervisor..."
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart "${APP_NAME}"

# === Step 5: Check Status ===
echo "Streamlit app '${APP_NAME}' status:"
sudo supervisorctl status "${APP_NAME}"


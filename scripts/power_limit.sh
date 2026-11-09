#!/usr/bin/env bash
# Cap the GPU's power (less heat, little speed lost) and keep the cap after reboots.
#   sudo bash scripts/power_limit.sh 200      # watts (V100 SXM2: 150-300, default 300)
#   sudo bash scripts/power_limit.sh off      # back to the card's default
set -euo pipefail
W="${1:-200}"
[ "$(id -u)" -eq 0 ] || { echo "Run with sudo."; exit 1; }
if [ "$W" = "off" ]; then
  systemctl disable --now nova-gpu-power.service 2>/dev/null || true
  rm -f /etc/systemd/system/nova-gpu-power.service; systemctl daemon-reload
  nvidia-smi -pl "$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)"
  echo "Power limit back to default."; exit 0
fi
MIN=$(nvidia-smi --query-gpu=power.min_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
MAX=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ "$W" -lt "$MIN" ] || [ "$W" -gt "$MAX" ]; then echo "This card allows $MIN-$MAX W."; exit 1; fi
cat > /etc/systemd/system/nova-gpu-power.service <<UNIT
[Unit]
Description=NOVA: cap GPU power at ${W} W (less heat)
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -pl ${W}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now nova-gpu-power.service
nvidia-smi --query-gpu=index,name,power.limit,temperature.gpu --format=csv
echo "Done: capped at ${W} W, also after reboots. Undo: sudo bash scripts/power_limit.sh off"

#!/bin/bash
# Isolated L4 window. Stops only voice+vllm, always restarts them.
# Caller: operator or agent via SSH, not imported by Python.
set -u
COMPOSE=/home/ubuntu/mysmartfood/infrastructure/scaleway-vllm/docker-compose.yml
LAB=/home/ubuntu/omni-s2s-lab
LOG=/tmp/omni-s2s-run.log
STATUS=/tmp/omni-s2s-window.status
export HF_HOME=/srv/mysmartfood/model-cache
export PYTHONUNBUFFERED=1

restore() {
  echo "RESTORE_START $(date -u +%FT%TZ)" | tee -a "$LOG"
  sudo docker compose -f "$COMPOSE" start vllm voice
  for _ in $(seq 1 18); do
    if sudo docker exec scaleway-vllm-voice-1 python3.11 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/live', timeout=5).read()" >/dev/null 2>&1; then
      echo "VOICE_UP $(date -u +%FT%TZ)" | tee -a "$LOG"
      break
    fi
    sleep 10
  done
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "$LOG"
  sudo docker ps --format "table {{.Names}}\t{{.Status}}" | tee -a "$LOG"
  echo "RESTORE_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
}

echo "WINDOW_START $(date -u +%FT%TZ)" > "$LOG"
echo running > "$STATUS"
if ! sudo docker compose -f "$COMPOSE" stop voice vllm; then
  echo STOP_FAILED | tee -a "$LOG"
  echo failed > "$STATUS"
  restore
  exit 1
fi
freed=0
for _ in $(seq 1 12); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d " ")
  echo "VRAM_USED_MIB=$used" | tee -a "$LOG"
  if [ "${used:-99999}" -lt 2000 ]; then
    freed=1
    break
  fi
  sleep 5
done
if [ "$freed" -ne 1 ]; then
  echo VRAM_NOT_FREE | tee -a "$LOG"
  echo failed > "$STATUS"
  restore
  exit 1
fi
set +e
cd "$LAB"
./venv/bin/python -m prototypes.omni_s2s.proto_s2s --force --reps 5 \
  --out "$LAB/benchmarks/results/omni_s2s_probe.json" \
  --wav-dir "$LAB/benchmarks/results/omni_s2s_wavs" >> "$LOG" 2>&1
rc=$?
echo "PROTO_EXIT=$rc $(date -u +%FT%TZ)" | tee -a "$LOG"
set -e
restore
if [ "$rc" -eq 0 ]; then echo ok > "$STATUS"; else echo failed > "$STATUS"; fi
exit "$rc"

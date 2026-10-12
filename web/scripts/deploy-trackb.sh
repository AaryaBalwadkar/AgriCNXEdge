#!/usr/bin/env bash
# Post-Ansible deploy for Track B test-and-drop. Run AFTER playbook.yml.
# Builds frontend LOCALLY (avoids OOM on 1GB VM), then ships code + model.
set -euo pipefail
VM_IP="${1:?usage: deploy-trackb.sh <VM_EPHEMERAL_IP> <GCE_USER>}"
GCE_USER="${2:?usage: deploy-trackb.sh <VM_EPHEMERAL_IP> <GCE_USER>}"
APP_DIR=/opt/agricnxedge-web

# 1. Frontend built locally (Windows: run `npm run build` in frontend\ first)
# 2. Ship backend code (gitignored model excluded) + local dist
tar --exclude='.venv' --exclude='__pycache__' -czf /tmp/agricnxedge-backend.tgz -C backend app requirements.txt pytest.ini
scp /tmp/agricnxedge-backend.tgz "$GCE_USER@$VM_IP:/tmp/"
scp backend/models/adc_student_full.onnx "$GCE_USER@$VM_IP:/tmp/adc_student_full.onnx"

# 3. Frontend dist built locally -> ship (run npm run build first)
if [ -d frontend/dist ]; then
  tar -czf /tmp/agricnxedge-dist.tgz -C frontend dist
  scp /tmp/agricnxedge-dist.tgz "$GCE_USER@$VM_IP:/tmp/"
else
  echo "frontend/dist missing: run 'npm run build' in frontend/ first" >&2
  exit 1
fi

ssh "$GCE_USER@$VM_IP" "sudo mkdir -p $APP_DIR/backend $APP_DIR/frontend && sudo chown -R agricnx:agricnx $APP_DIR"
ssh "$GCE_USER@$VM_IP" "sudo -u agricnx tar -xzf /tmp/agricnxedge-backend.tgz -C $APP_DIR/backend && sudo -u agricnx rm -rf $APP_DIR/frontend/dist && sudo -u agricnx mkdir -p $APP_DIR/frontend && sudo -u agricnx tar -xzf /tmp/agricnxedge-dist.tgz -C $APP_DIR/frontend && sudo -u agricnx cp /tmp/adc_student_full.onnx $APP_DIR/backend/models/adc_student_full.onnx"
ssh "$GCE_USER@$VM_IP" "cd $APP_DIR/backend && (sudo -u agricnx python3 -m venv .venv || true) && sudo -u agricnx $APP_DIR/backend/.venv/bin/pip install -q -r requirements.txt"
ssh "$GCE_USER@$VM_IP" "sudo systemctl restart agricnxedge-api && sleep 5 && curl -sf http://127.0.0.1:8000/api/health && curl -sf http://127.0.0.1:8000/api/model/status"
echo "Verify from your laptop: http://$VM_IP/ and http://$VM_IP/api/health"

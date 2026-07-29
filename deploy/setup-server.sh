#!/usr/bin/env bash
#
# One-time server preparation. Run once on a fresh VPS, as root or with sudo.
#
#   curl -fsSL https://raw.githubusercontent.com/akobirbarotovdev/multi-agent-ai-system/main/deploy/setup-server.sh | bash
#
# or, from a checkout:
#
#   sudo ./deploy/setup-server.sh
#
# This never writes secrets. It creates the env file skeleton with mode 600 and
# leaves the values for you to fill in.

set -euo pipefail

APP_DIR="${APP_DIR:-/opt/multi-agent-ai-system}"
REPO_URL="${REPO_URL:-https://github.com/akobirbarotovdev/multi-agent-ai-system.git}"
DEPLOY_USER="${DEPLOY_USER:-deploy}"

log() { printf '\033[0;36m[setup]\033[0m %s\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "run as root or with sudo" >&2; exit 1; }

# ── Docker ────────────────────────────────────────────────
if ! command -v docker >/dev/null; then
    log "Installing Docker"
    curl -fsSL https://get.docker.com | sh
fi
docker compose version >/dev/null 2>&1 || {
    echo "the docker compose plugin is missing; install docker-compose-plugin" >&2
    exit 1
}

# ── Deploy user ───────────────────────────────────────────
if ! id "$DEPLOY_USER" >/dev/null 2>&1; then
    log "Creating the $DEPLOY_USER user"
    useradd --create-home --shell /bin/bash "$DEPLOY_USER"
fi
usermod -aG docker "$DEPLOY_USER"

# ── Checkout ──────────────────────────────────────────────
if [ ! -d "$APP_DIR/.git" ]; then
    log "Cloning into $APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
fi
chown -R "$DEPLOY_USER:$DEPLOY_USER" "$APP_DIR"
chmod +x "$APP_DIR/deploy/deploy.sh"

# ── Env file skeleton (no secrets written) ────────────────
ENV_FILE="$APP_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    log "Creating $ENV_FILE — fill in the values before the first deploy"
    cat > "$ENV_FILE" <<'EOF'
# Production configuration. Never commit this file.
APP_ENV=production
DEMO_MODE=false
LOG_FORMAT=json
LOG_LEVEL=INFO

# Required — the app refuses to start in production without these.
ANTHROPIC_API_KEY=
# Generate with: openssl rand -hex 32
ADMIN_API_KEY=
# Comma-separated browser origins. "*" is rejected in production.
CORS_ORIGINS=https://your-domain.example

# Optional
ROUTER_MODEL=claude-sonnet-4-6
AGENT_MODEL=claude-sonnet-4-6
TWOCHAT_WEBHOOK_SECRET=
TELEGRAM_BOT_TOKEN=
LANGSMITH_API_KEY=
# Drain the follow-up queue every 15 minutes. Enable on exactly one replica.
FOLLOWUP_INTERVAL_SECONDS=900
EOF
    chown "$DEPLOY_USER:$DEPLOY_USER" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
else
    log "$ENV_FILE already exists — leaving it untouched"
fi

log "Done. Next:"
echo "  1. Fill in $ENV_FILE (ANTHROPIC_API_KEY, ADMIN_API_KEY, CORS_ORIGINS)"
echo "  2. Add the $DEPLOY_USER user's public key to ~$DEPLOY_USER/.ssh/authorized_keys"
echo "  3. Put a TLS-terminating reverse proxy in front of 127.0.0.1:8000"
echo "  4. Deploy:  sudo -u $DEPLOY_USER $APP_DIR/deploy/deploy.sh origin/main"

#!/usr/bin/env bash
#
# Deploy the Multi-Agent AI System on a single VPS.
#
# Runs ON the server. Safe to re-run; safe to run against a host that is
# already serving traffic.
#
#   ./deploy/deploy.sh <git-ref>
#
# The ordering matters: every check that can fail is done BEFORE the running
# container is touched, so a bad deploy leaves the current version serving.
# Version 1.1 added required production settings (ADMIN_API_KEY, CORS_ORIGINS);
# an env file written for an earlier release will not satisfy them, and a naive
# restart would take the service down. The preflight below catches that while
# the old container is still up.

set -euo pipefail

APP_DIR="${APP_DIR:-/opt/multi-agent-ai-system}"
ENV_FILE="${ENV_FILE:-$APP_DIR/.env}"
COMPOSE_FILE="$APP_DIR/deploy/docker-compose.prod.yml"
BIND_PORT="${BIND_PORT:-8000}"
HEALTH_URL="http://127.0.0.1:${BIND_PORT}/health/ready"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-90}"
TARGET_REF="${1:-origin/main}"

log()  { printf '\033[0;36m[deploy]\033[0m %s\n' "$*"; }
ok()   { printf '\033[0;32m[  ok  ]\033[0m %s\n' "$*"; }
fail() { printf '\033[0;31m[ fail ]\033[0m %s\n' "$*" >&2; exit 1; }

compose() { docker compose -f "$COMPOSE_FILE" --project-directory "$APP_DIR" "$@"; }

# ── 0. Prerequisites ─────────────────────────────────────
command -v docker >/dev/null || fail "docker is not installed"
docker compose version >/dev/null 2>&1 || fail "the docker compose plugin is not installed"
command -v curl >/dev/null || fail "curl is not installed"
[ -d "$APP_DIR/.git" ] || fail "$APP_DIR is not a git checkout — run the one-time setup first"

cd "$APP_DIR"

# ── 1. Fetch the target revision (nothing is switched yet) ─
log "Fetching $TARGET_REF"
git fetch --prune origin
PREVIOUS_SHA="$(git rev-parse HEAD)"
TARGET_SHA="$(git rev-parse "$TARGET_REF")"
log "Current  : $PREVIOUS_SHA"
log "Deploying: $TARGET_SHA"

if [ "$PREVIOUS_SHA" = "$TARGET_SHA" ]; then
    log "Already at the target revision; continuing so config changes still apply."
fi

# ── 2. Preflight the operator-managed env file ────────────
#
# Values are never printed — only the presence of each key is reported.
[ -f "$ENV_FILE" ] || fail "env file not found at $ENV_FILE"

perms="$(stat -c '%a' "$ENV_FILE")"
case "$perms" in
    600|400) ;;
    *) log "WARNING: $ENV_FILE is mode $perms; 600 is recommended" ;;
esac

has_var() {
    # Matches KEY=<something non-empty>, ignoring comments and whitespace.
    grep -Eq "^[[:space:]]*${1}[[:space:]]*=[[:space:]]*[^[:space:]#].*$" "$ENV_FILE"
}

REQUIRED=(ANTHROPIC_API_KEY ADMIN_API_KEY CORS_ORIGINS)
MISSING=()
for var in "${REQUIRED[@]}"; do
    has_var "$var" || MISSING+=("$var")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    cat >&2 <<EOF

[ fail ] $ENV_FILE is missing required production settings:

$(printf '           - %s\n' "${MISSING[@]}")

  Release 1.1 refuses to start in production without these, so deploying now
  would stop the service. The running container has NOT been touched — it is
  still serving on the previous version.

  Add the missing keys, then re-run this script:

    ADMIN_API_KEY   a shared secret guarding the operator dashboard, CRM reads
                    and knowledge-base writes. Generate: openssl rand -hex 32
    CORS_ORIGINS    comma-separated browser origins, e.g.
                    https://app.example.com  ("*" is rejected in production)

EOF
    exit 1
fi

if has_var DEMO_MODE && grep -Eqi '^[[:space:]]*DEMO_MODE[[:space:]]*=[[:space:]]*(true|1|yes|on)' "$ENV_FILE"; then
    fail "DEMO_MODE is enabled in $ENV_FILE — refusing to deploy mock responses to production"
fi

if grep -Eq '^[[:space:]]*CORS_ORIGINS[[:space:]]*=[[:space:]]*\*[[:space:]]*$' "$ENV_FILE"; then
    fail "CORS_ORIGINS is '*' in $ENV_FILE — set an explicit origin list"
fi

ok "Preflight passed (${#REQUIRED[@]} required settings present)"

# ── 3. Check out and build (old container still serving) ──
log "Checking out $TARGET_SHA"
git checkout --quiet --detach "$TARGET_SHA"

log "Building image"
if ! compose build; then
    log "Build failed — restoring $PREVIOUS_SHA; the running container was never stopped"
    git checkout --quiet --detach "$PREVIOUS_SHA"
    fail "docker compose build failed"
fi
ok "Image built"

# ── 4. Switch over ────────────────────────────────────────
rollback() {
    log "Rolling back to $PREVIOUS_SHA"
    git checkout --quiet --detach "$PREVIOUS_SHA"
    if compose up -d --build; then
        log "Rollback complete — previous version is serving again"
    else
        printf '\033[0;31m[ fail ]\033[0m Rollback FAILED. Service is down. Inspect: docker compose -f %s logs\n' "$COMPOSE_FILE" >&2
    fi
}

log "Starting the new container"
if ! compose up -d --remove-orphans; then
    rollback
    fail "docker compose up failed"
fi

# ── 5. Verify health before declaring success ─────────────
log "Waiting for /health/ready (timeout ${HEALTH_TIMEOUT}s)"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
healthy=false
while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
        healthy=true
        break
    fi
    sleep 3
done

if [ "$healthy" != true ]; then
    log "New version did not become ready. Recent logs:"
    compose logs --tail 60 app >&2 || true
    rollback
    fail "health check failed after ${HEALTH_TIMEOUT}s"
fi

ok "Healthy: $(curl -fsS --max-time 5 "$HEALTH_URL")"

# ── 6. Tidy up ────────────────────────────────────────────
docker image prune -f --filter "until=168h" >/dev/null 2>&1 || true

ok "Deployed $TARGET_SHA (previous: $PREVIOUS_SHA)"

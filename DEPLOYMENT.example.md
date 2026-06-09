# Local deployment notes — NEVER commit this file.
# Copy to .env.local on your laptop and source it before running deploy scripts:
#   cp DEPLOYMENT.example.md .env.local
#   # edit, then:
#   source .env.local
#   GATEWAY_AUTH_TOKEN="$TOKEN" bash scripts/deploy_h20.sh

# H20 box (vllm-omini + gateway co-located)
export REMOTE="root@<your-h20-ip>"
export REMOTE_HOST="<your-h20-ip>"

# Bookkeeping — what was deployed when, who has the token.
# DEPLOYED_AT=2026-06-09
# TOKEN_FINGERPRINT=$(echo -n "$TOKEN" | sha256sum | head -c 12)
# SHARED_WITH="alice@team, bob@team"

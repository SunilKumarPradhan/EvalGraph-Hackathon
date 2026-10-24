#!/bin/sh
# EvalGraph sandbox entrypoint. Runs as root only long enough to:
#   1. move the DIAL credentials into a root-only file and out of the environment,
#   2. start the LLM gateway (root),
#   3. hand over to the harness as the unprivileged `app` user with a clean environment.
set -eu

mkdir -p /run/evalgraph
chmod 700 /run/evalgraph
umask 077
python3 - <<'PY'
import json, os
json.dump({
    "endpoint": os.environ.get("DIAL_ENDPOINT", ""),
    "api_key": os.environ.get("DIAL_API_KEY", ""),
    "api_version": os.environ.get("DIAL_API_VERSION", "2024-02-01"),
    "chat_model": os.environ.get("SANDBOX_CHAT_MODEL", "gpt-4.1-mini-2025-04-14"),
    "embed_model": os.environ.get("SANDBOX_EMBED_MODEL", "text-embedding-3-small-1"),
    "mock": os.environ.get("SANDBOX_MOCK_LLM", "0") == "1",
    "honor_model": os.environ.get("SANDBOX_HONOR_MODEL", "1") == "1",
}, open("/run/evalgraph/secrets.json", "w"))
PY
unset DIAL_API_KEY DIAL_ENDPOINT DIAL_API_VERSION

# user-provided app secrets (APPSECRET_<NAME>): for the app only, never for the gateway's DIAL key
mkdir -p /run/evalgraph-app
python3 - <<'PY'
import json, os
secrets = {k[len("APPSECRET_"):]: v for k, v in os.environ.items() if k.startswith("APPSECRET_")}
json.dump(secrets, open("/run/evalgraph-app/secrets.json", "w"))
PY
for v in $(env | sed -n 's/^\(APPSECRET_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
chown -R app:app /run/evalgraph-app
chmod 500 /run/evalgraph-app && chmod 400 /run/evalgraph-app/secrets.json

PROXY_ENV=""
if [ -n "${EVALGRAPH_PROXY:-}" ]; then  # egress allowlist: the only way out is the proxy container
  export HTTP_PROXY="$EVALGRAPH_PROXY" HTTPS_PROXY="$EVALGRAPH_PROXY" http_proxy="$EVALGRAPH_PROXY" https_proxy="$EVALGRAPH_PROXY"
  export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
  PROXY_ENV="HTTP_PROXY=$EVALGRAPH_PROXY HTTPS_PROXY=$EVALGRAPH_PROXY http_proxy=$EVALGRAPH_PROXY https_proxy=$EVALGRAPH_PROXY NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost npm_config_proxy=$EVALGRAPH_PROXY npm_config_https_proxy=$EVALGRAPH_PROXY"
fi

EVALGRAPH_SECRET_FILE=/run/evalgraph/secrets.json EVALGRAPH_LLM_LOG=/out/llm_calls.jsonl \
  python3 /harness/gateway.py 2>>/out/gateway.log &

mkdir -p /work
chown app:app /work /out
CACHE_ENV=""
if [ -d /cache ]; then  # per-repo download cache (pip wheels, npm, model files), a Docker volume
  mkdir -p /cache/pip /cache/npm /cache/hf /cache/xdg /cache/fastembed
  chown app:app /cache /cache/pip /cache/npm /cache/hf /cache/xdg /cache/fastembed
  CACHE_ENV="PIP_CACHE_DIR=/cache/pip npm_config_cache=/cache/npm HF_HOME=/cache/hf XDG_CACHE_HOME=/cache/xdg FASTEMBED_CACHE_PATH=/cache/fastembed"
fi
# shellcheck disable=SC2086
exec runuser -u app -- env -i HOME=/home/app PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 $CACHE_ENV $PROXY_ENV \
  python3 /harness/harness.py /out/job.json

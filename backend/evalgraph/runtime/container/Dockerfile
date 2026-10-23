# EvalGraph sandbox: clones, installs and runs an analyzed repo in isolation.
# Built automatically by evalgraph.runtime.sandbox (tag evalgraph-sandbox:<hash of this folder>).
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git nodejs npm build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app \
 && mkdir -p /work /out /harness

COPY harness/ /harness/
COPY entrypoint.sh /entrypoint.sh

# the gateway and harness are stdlib-only; nothing from EvalGraph itself is inside the image
ENTRYPOINT ["/bin/sh", "/entrypoint.sh"]

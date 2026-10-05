#!/usr/bin/env bash
set -euo pipefail

docker run -d \
    --name friday-postgres \
    --restart unless-stopped \
    -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD:?set POSTGRES_PASSWORD}" \
    -e POSTGRES_USER=friday \
    -e POSTGRES_DB=friday \
    -p "0.0.0.0:6464:5432" \
    postgres:17

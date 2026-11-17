#!/usr/bin/env bash
# Hold a logind sleep/idle inhibitor while this experiment's service runs.
# Kept outside the frozen model/training Python sources.
set -euo pipefail

if [[ "${1:-}" != "--watch" ]]; then
    exec /usr/bin/systemd-inhibit --no-ask-password \
        --what=sleep:idle --mode=block --who="LiMT training" \
        --why="Keep the computer awake until the joint prediction experiment finishes" \
        /usr/bin/bash "$0" --watch "${1:-limt-joint-price-volatility.service}"
fi

target_unit="${2:?A training service is required}"
while true; do
    target_state="$(systemctl --user show "$target_unit" --property=ActiveState --value)"
    case "$target_state" in
        active|activating|reloading|deactivating) ;;
        inactive|failed) exit 0 ;;
        *) exit 1 ;;
    esac
    sleep 10
done

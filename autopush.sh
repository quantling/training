#!/bin/bash

# Pfad zum Git-Repository
REPO_PATH="./"
SLEEP_DAYS=3
SLEEP_SECONDS=$((SLEEP_DAYS * 24 * 60 * 60))

while true; do
    cd "$REPO_PATH" || exit 1

    # Füge alle Änderungen hinzu
    git add .

    # Commit nur, wenn es Änderungen gibt
    if ! git diff --cached --quiet; then
        git commit -m "push new losses (automated)"
        git push origin main
    else
        echo "[$(date)] Keine Änderungen zum Committen"
    fi

    # Warte 3 Tage
    sleep "$SLEEP_SECONDS"
done

#!/bin/bash
# Install git hooks from the hooks/ directory.
# Run from project root: bash scripts/setup_hooks.sh

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="$REPO_ROOT/hooks"
HOOKS_DST="$REPO_ROOT/.git/hooks"

if [ ! -d "$HOOKS_SRC" ]; then
    echo "No hooks/ directory found at $HOOKS_SRC"
    exit 1
fi

for hook in "$HOOKS_SRC"/*; do
    hook_name="$(basename "$hook")"
    cp "$hook" "$HOOKS_DST/$hook_name"
    chmod +x "$HOOKS_DST/$hook_name"
    echo "Installed $hook_name"
done

echo "All hooks installed."

#!/bin/bash
set -euo pipefail

# diff.sh - Generates a markdown diff of all changes relative to origin/main (or specified branch)

OUTPUT="diff_report.md"
BRANCH=${1:-"origin/main"}

echo "🔍 Fetching latest from remote..."
git fetch origin

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "🌿 Current Branch: $CURRENT_BRANCH"
echo "🎯 Comparing against: $BRANCH"

echo "# Diff Report: $CURRENT_BRANCH vs $BRANCH" > $OUTPUT
echo "**Date:** $(date)" >> $OUTPUT
echo "" >> $OUTPUT

# 1. File Summary
echo "## 📂 Changed Files" >> $OUTPUT
echo '```' >> $OUTPUT
git diff --name-status $BRANCH >> $OUTPUT || echo "No changes found." >> $OUTPUT
echo '```' >> $OUTPUT
echo "" >> $OUTPUT

# 2. Detailed Diff
echo "## 📝 Detailed Changes" >> $OUTPUT
echo "" >> $OUTPUT

# Get list of changed text files (exclude binary)
# We assume standard diff excludes binary output by default or shows "Binary file matches"
# We wrap the whole diff block.
echo '```diff' >> $OUTPUT
git diff $BRANCH >> $OUTPUT || true
echo '```' >> $OUTPUT

echo "✅ Diff written to $OUTPUT"
echo "   (You can open this file to see what has changed since the last push/fetch)"

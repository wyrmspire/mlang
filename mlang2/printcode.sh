#!/bin/bash
# =============================================================================
# printcode.sh - Dump project code to markdown files
# =============================================================================
# 
# Outputs project structure and code to dump1.md, dump2.md, etc.
# Each file contains ~1000 lines.
#
# Excludes:
#   - __pycache__
#   - .git
#   - node_modules
#   - dist, .next, build
#   - data/ (raw data files)
#   - cache/
#   - shards/
#   - models/ (trained weights)
#   - results/
#   - *.parquet, *.pth, *.json (data files)
#   - *.pyc, *.lock
#
# Usage: ./printcode.sh
# =============================================================================

set -e

OUTPUT_PREFIX="dump"
LINES_PER_FILE=1000
TEMP_FILE=$(mktemp)

# Project root (where this script lives)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "# MLang2 Project Code Dump" > "$TEMP_FILE"
echo "Generated: $(date)" >> "$TEMP_FILE"
echo "" >> "$TEMP_FILE"

# Common excludes
EXCLUDES=(
    "*/__pycache__/*"
    "*/.git/*"
    "*/node_modules/*"
    "*/dist/*"
    "*/.next/*"
    "*/build/*"
    "*/data/*"
    "*/cache/*"
    "*/shards/*"
    "*/results/*"
    "*/.venv/*"
    "*/venv/*"
)

# Project structure
echo "## Project Structure" >> "$TEMP_FILE"
echo '```' >> "$TEMP_FILE"
find "$PROJECT_ROOT" -type f \
    ! -path "*/__pycache__/*" \
    ! -path "*/.git/*" \
    ! -path "*/node_modules/*" \
    ! -path "*/dist/*" \
    ! -path "*/.next/*" \
    ! -path "*/build/*" \
    ! -path "*/data/*" \
    ! -path "*/cache/*" \
    ! -path "*/shards/*" \
    ! -path "*/models/*.pth" \
    ! -path "*/results/*" \
    ! -path "*/.venv/*" \
    ! -path "*/venv/*" \
    ! -name "*.pyc" \
    ! -name "*.parquet" \
    ! -name "*.pth" \
    ! -name "*.lock" \
    ! -name "package-lock.json" \
    ! -name "continuous_contract.json" \
    ! -name "dump*.md" \
    ! -name "dump*" \
    | sed "s|$PROJECT_ROOT/||" \
    | sort >> "$TEMP_FILE"
echo '```' >> "$TEMP_FILE"
echo "" >> "$TEMP_FILE"

# Collect all code files
echo "## Source Files" >> "$TEMP_FILE"
echo "" >> "$TEMP_FILE"

find "$PROJECT_ROOT" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" -o -name "*.ts" -o -name "*.tsx" -o -name "*.css" \) \
    ! -path "*/__pycache__/*" \
    ! -path "*/.git/*" \
    ! -path "*/node_modules/*" \
    ! -path "*/dist/*" \
    ! -path "*/.next/*" \
    ! -path "*/build/*" \
    ! -path "*/data/*" \
    ! -path "*/cache/*" \
    ! -path "*/shards/*" \
    ! -path "*/results/*" \
    ! -path "*/.venv/*" \
    ! -path "*/venv/*" \
    ! -name "dump*.md" \
    | sort | while read -r file; do
    
    rel_path="${file#$PROJECT_ROOT/}"
    ext="${file##*.}"
    
    echo "### $rel_path" >> "$TEMP_FILE"
    echo "" >> "$TEMP_FILE"
    
    # Determine language for syntax highlighting
    case "$ext" in
        py) lang="python" ;;
        sh) lang="bash" ;;
        md) lang="markdown" ;;
        yaml|yml) lang="yaml" ;;
        ts) lang="typescript" ;;
        tsx) lang="tsx" ;;
        css) lang="css" ;;
        *) lang="" ;;
    esac
    
    echo "\`\`\`$lang" >> "$TEMP_FILE"
    cat "$file" >> "$TEMP_FILE"
    echo "" >> "$TEMP_FILE"
    echo "\`\`\`" >> "$TEMP_FILE"
    echo "" >> "$TEMP_FILE"
done

# Split into chunks
total_lines=$(wc -l < "$TEMP_FILE")
num_files=$(( (total_lines + LINES_PER_FILE - 1) / LINES_PER_FILE ))

echo "Total lines: $total_lines"
echo "Splitting into $num_files files..."

# Remove old dump files
rm -f "$PROJECT_ROOT"/${OUTPUT_PREFIX}*.md
rm -f "$PROJECT_ROOT"/${OUTPUT_PREFIX}[0-9]*

# Split (use 2-digit suffix to handle more files)
split -l $LINES_PER_FILE -d -a 2 "$TEMP_FILE" "$PROJECT_ROOT/${OUTPUT_PREFIX}"

# Rename to .md
for f in "$PROJECT_ROOT"/${OUTPUT_PREFIX}*; do
    if [[ ! "$f" =~ \.md$ ]]; then
        mv "$f" "${f}.md"
    fi
done

# Cleanup
rm -f "$TEMP_FILE"

echo "Done! Created:"
ls -la "$PROJECT_ROOT"/${OUTPUT_PREFIX}*.md

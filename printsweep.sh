#!/bin/bash

OUTPUT="sweep_codebase.md"
echo "# Sweep Codebase and Documentation" > "$OUTPUT"
echo "Generated on $(date)" >> "$OUTPUT"
echo "" >> "$OUTPUT"

print_file() {
    FILE_PATH="$1"
    LANG="$2"
    
    if [ -f "$FILE_PATH" ]; then
        echo "Adding $FILE_PATH..."
        echo "## File: $FILE_PATH" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
        echo "\`\`\`$LANG" >> "$OUTPUT"
        cat "$FILE_PATH" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
        echo "\`\`\`" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
        echo "---" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
    else
        echo "Warning: File $FILE_PATH not found, skipping."
    fi
}

echo "Generating $OUTPUT..."

# Core Sweep Pipeline
print_file "src/sweep/supersweep.py" "python"
print_file "src/sweep/pattern_miner_v2.py" "python"
print_file "src/sweep/train_sweep.py" "python"
print_file "src/sweep/oco_tester.py" "python"
print_file "src/sweep/param_grid.py" "python"
print_file "src/sweep/config.py" "python"
print_file "src/sweep/run_sweep.py" "python"

# Documentation
print_file "docs/sweep_master_guide.md" "markdown"
print_file "docs/best_config.md" "markdown"
print_file "docs/success_study.md" "markdown"

echo "Done! Output saved to $OUTPUT"

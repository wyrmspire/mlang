#!/bin/bash

# Generates a Markdown dump of the project structure and code.

OUTPUT="project_code.md"

echo "# Project Code Dump" > "$OUTPUT"
echo "" >> "$OUTPUT"
echo "## Project Structure" >> "$OUTPUT"
echo '```' >> "$OUTPUT"
# Simulated tree using find, ignoring excluded dirs
find . -print | grep -v -E "\.git/|node_modules/|__pycache__/|logs/|\.gemini/|dist/|build/|\.venv" | sed -e 's;[^/]*/;|____;g;s;____|; |;g' >> "$OUTPUT"
echo '```' >> "$OUTPUT"

echo "" >> "$OUTPUT"
echo "## File Contents" >> "$OUTPUT"

# Find interesting files in src and frontend (if exists), excluding common noise
find . \
    -type d \( -name ".git" -o -name "node_modules" -o -name "__pycache__" -o -name "logs" -o -name "data" -o -name ".gemini" -o -name "dist" -o -name "build" -o -name ".venv312" \) -prune -o \
    -type f \( -name "*.py" -o -name "*.ts" -o -name "*.tsx" -o -name "package.json" -o -name "requirements.txt" \) \
    -print0 | while IFS= read -r -d '' file; do
        
        # Skip this script itself and the output file
        if [[ "$file" == "./printcode.sh" ]]; then continue; fi
        if [[ "$file" == "./$OUTPUT" ]]; then continue; fi

        echo "" >> "$OUTPUT"
        echo "### $file" >> "$OUTPUT"
        
        # Detect extension for code fence
        EXT="${file##*.}"
        echo '```'"$EXT" >> "$OUTPUT"
        cat "$file" >> "$OUTPUT"
        echo '```' >> "$OUTPUT"
done

echo "Done! Code dumped to $OUTPUT"
cat "$OUTPUT"

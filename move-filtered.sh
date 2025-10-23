#!/bin/bash

# Check if a file path parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <text_file_path>"
    echo "Example: $0 file_patterns.txt"
    exit 1
fi

# Get the text file path from the first parameter
TEXT_FILE="$1"

# Check if the text file exists
if [ ! -f "$TEXT_FILE" ]; then
    echo "Error: File '$TEXT_FILE' does not exist."
    exit 1
fi

# Get the current executing directory
CURRENT_DIR=$(pwd)
echo "Current directory: $CURRENT_DIR"

# Create 'filtered' directory if it doesn't exist
if [ ! -d "filtered" ]; then
    mkdir -p filtered
    echo "Created 'filtered' directory"
else
    echo "'filtered' directory already exists"
fi

# Read each line from the text file and process it
while IFS= read -r line; do
    # Skip empty lines
    if [ -z "$line" ]; then
        continue
    fi
    
    # Remove any leading/trailing whitespace
    line=$(echo "$line" | xargs)
    
    echo "Processing pattern: $line"
    
    # Move files matching the pattern from all subdirectories to filtered
    # The pattern is: ./*/{linecontent}*
    mv ./*/"$line"* filtered/    
done < "$TEXT_FILE"

echo "File moving completed!"

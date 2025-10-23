#!/bin/bash

# Check if both origin and target directory parameters are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <origin_directory> <target_directory>"
    echo "Example: $0 /path/to/source /path/to/destination"
    exit 1
fi

# Get the origin and target directories from parameters
ORIGIN_DIR="$1"
TARGET_DIR="$2"

# Check if origin directory exists
if [ ! -d "$ORIGIN_DIR" ]; then
    echo "Error: Origin directory '$ORIGIN_DIR' does not exist."
    exit 1
fi

# Create target directory if it doesn't exist
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
    echo "Created target directory: $TARGET_DIR"
fi

# Create test and valid directories in target
mkdir -p "$TARGET_DIR/test/images"
mkdir -p "$TARGET_DIR/test/labels"
mkdir -p "$TARGET_DIR/valid/images"
mkdir -p "$TARGET_DIR/valid/labels"

echo "Created consolidated directories in: $TARGET_DIR"

# Function to copy files from source to target
copy_files() {
    local source_dir="$1"
    local target_images_dir="$2"
    local target_labels_dir="$3"
    local dir_type="$4"
    
    if [ -d "$source_dir/images" ]; then
        echo "Copying images from $source_dir/images to $target_images_dir"
        cp -r "$source_dir/images"/* "$target_images_dir"/ 2>/dev/null || true
    fi
    
    if [ -d "$source_dir/labels" ]; then
        echo "Copying labels from $source_dir/labels to $target_labels_dir"
        cp -r "$source_dir/labels"/* "$target_labels_dir"/ 2>/dev/null || true
    fi
}

# Counter for tracking found directories
test_count=0
valid_count=0

echo "Searching for test and valid directories in: $ORIGIN_DIR"

# Find all test directories recursively
while IFS= read -r -d '' test_dir; do
    test_count=$((test_count + 1))
    echo "Found test directory: $test_dir"
    copy_files "$test_dir" "$TARGET_DIR/test/images" "$TARGET_DIR/test/labels" "test"
done < <(find "$ORIGIN_DIR" -type d -name "test" -print0)

# Find all valid directories recursively
while IFS= read -r -d '' valid_dir; do
    valid_count=$((valid_count + 1))
    echo "Found valid directory: $valid_dir"
    copy_files "$valid_dir" "$TARGET_DIR/valid/images" "$TARGET_DIR/valid/labels" "valid"
done < <(find "$ORIGIN_DIR" -type d -name "valid" -print0)

echo ""
echo "Consolidation completed!"
echo "Found and processed:"
echo "  - $test_count test directories"
echo "  - $valid_count valid directories"
echo ""
echo "Consolidated structure in $TARGET_DIR:"
echo "  - test/images/ (contains all images from test directories)"
echo "  - test/labels/ (contains all labels from test directories)"
echo "  - valid/images/ (contains all images from valid directories)"
echo "  - valid/labels/ (contains all labels from valid directories)"

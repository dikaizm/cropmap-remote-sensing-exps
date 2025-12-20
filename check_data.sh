#!/bin/bash
# Script to check processed data on RunPod
# Run this on the RunPod instance: bash check_data.sh

echo "=== Checking Processed Data Directory ==="
DATA_DIR="$HOME/crop_mapping_pipeline/data/processed"
echo "Location: $DATA_DIR"
echo ""

echo "=== S2 Processed Data ==="
for year in 2022 2023 2024; do
    echo "--- Year $year ---"
    S2_DIR="$DATA_DIR/s2/$year"
    if [ -d "$S2_DIR" ]; then
        COUNT=$(ls -1 "$S2_DIR"/*_processed.tif 2>/dev/null | wc -l)
        echo "  Files: $COUNT"
        echo "  Size: $(du -sh "$S2_DIR" 2>/dev/null | cut -f1)"
        ls -lh "$S2_DIR"/*_processed.tif 2>/dev/null | head -3
        if [ $COUNT -gt 3 ]; then echo "  ... ($((COUNT-3)) more files)"; fi
    else
        echo "  Directory not found!"
    fi
    echo ""
done

echo "=== CDL Processed Data ==="
CDL_DIR="$DATA_DIR/cdl"
if [ -d "$CDL_DIR" ]; then
    echo "Location: $CDL_DIR"
    ls -lh "$CDL_DIR"/cdl_*_study_area_filtered.tif 2>/dev/null
    echo ""
    echo "File sizes:"
    du -sh "$CDL_DIR"/*_filtered.tif 2>/dev/null
else
    echo "CDL directory not found!"
fi
echo ""

echo "=== Stage 2/3 Handoff Files ==="
for file in stage2v2_per_crop_results.csv stage3_exp_c_bands.txt stage3_exp_c_bands_projected.json; do
    if [ -f "$DATA_DIR/$file" ]; then
        echo "✓ $file"
        ls -lh "$DATA_DIR/$file"
    else
        echo "✗ $file (not found)"
    fi
done

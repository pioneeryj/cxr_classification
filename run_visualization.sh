#!/bin/bash

# Experiment Results Visualization Runner
# Usage: ./run_visualization.sh [experiment_dir]
# Example: ./run_visualization.sh outputs/resnet50_experiment

EXPERIMENT_DIR=${1:-"outputs/resnet50_experiment"}
OUTPUT_DIR="${EXPERIMENT_DIR}/visualizations"
echo "Running visualization for: $EXPERIMENT_DIR"
python visualize_experiment_results.py --experiment_dir "$EXPERIMENT_DIR" --output_dir "$OUTPUT_DIR"

#!/bin/bash

# Run PI experiments for all AWS Bedrock models
# Usage: ./run_all_bedrock_models_pi.sh [num_runs]
# Example: ./run_all_bedrock_models_pi.sh 3  # Run each model 3 times

set -e  # Exit on error

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Number of runs per model (default: 1, use 3-5 for confidence intervals)
NUM_RUNS=${1:-1}

# Interference levels to test
LEVELS="3 10 50 100 200 300"

# Project root (parent of scripts/shell)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Add project root to PYTHONPATH for imports
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# List of all Bedrock models to test
MODELS=(
    # Meta Llama 3 (8K context)
    "bedrock-llama-3-8b"
    "bedrock-llama-3-70b"

    # Meta Llama 3.1 (128K context)
    "bedrock-llama-3.1-8b"
    "bedrock-llama-3.1-70b"
    "bedrock-llama-3.1-405b"

    # Meta Llama 3.2 (128K context)
    "bedrock-llama-3.2-1b"
    "bedrock-llama-3.2-3b"
    "bedrock-llama-3.2-11b"
    "bedrock-llama-3.2-90b"

    # Meta Llama 3.3 (128K context)
    "bedrock-llama-3.3-70b"

    # Meta Llama 4
    "bedrock-llama-4-scout-17b"
    "bedrock-llama-4-maverick-17b"

    # Amazon Titan
    "bedrock-titan-large"
    "bedrock-titan-lite"
    "bedrock-titan-express"

    # Amazon Nova
    "bedrock-nova-micro"
    "bedrock-nova-lite"
    "bedrock-nova-pro"
    "bedrock-nova-premier"

    # Mistral
    "bedrock-mistral-7b"
    "bedrock-mistral-8x7b"
    "bedrock-mistral-large"
    "bedrock-mistral-small"
    "bedrock-ministral-3b"
    "bedrock-ministral-8b"

    # DeepSeek
    "bedrock-deepseek-r1"

    # Qwen
    "bedrock-qwen-3-32b"
    "bedrock-qwen-3-vl-235b"
    "bedrock-qwen-3-coder-30b"
)

# Track progress
TOTAL=${#MODELS[@]}
CURRENT=0
FAILED=()
SUCCESS=()

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}Starting Bedrock Model PI Experiments${NC}"
echo -e "${BLUE}Testing ${TOTAL} models with levels: ${LEVELS}${NC}"
echo -e "${BLUE}Runs per model: ${NUM_RUNS}${NC}"
echo -e "${BLUE}Output directory: ${PROJECT_ROOT}/data/raw_experiments_pi/${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# Create output directory
mkdir -p data/raw_experiments_pi

# Note: Bedrock uses AWS credentials from ~/.aws/credentials
echo -e "${YELLOW}Note: Bedrock uses AWS credentials from ~/.aws/credentials${NC}"
echo ""

# Loop through each model
for model in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))

    echo ""
    echo -e "${GREEN}[${CURRENT}/${TOTAL}] Running PI experiment for: ${model}${NC}"
    echo -e "${GREEN}=====================================================================${NC}"

    # Run multiple times if requested
    MODEL_FAILED=false
    for run in $(seq 1 $NUM_RUNS); do
        if [ $NUM_RUNS -gt 1 ]; then
            echo -e "${YELLOW}Run ${run}/${NUM_RUNS}${NC}"
        fi

        # Run the PI experiment from project root
        if python scripts/core/run_proactive_interference_experiment.py \
            --model "${model}" \
            --levels ${LEVELS} \
            --output-dir data/raw_experiments_pi; then
            echo -e "${GREEN}✓ Successfully completed run ${run}: ${model}${NC}"
        else
            echo -e "${RED}✗ Failed run ${run}: ${model}${NC}"
            MODEL_FAILED=true
            break  # Don't continue runs if one fails
        fi

        # Small delay between runs
        if [ $run -lt $NUM_RUNS ]; then
            sleep 5
        fi
    done

    # Track success/failure
    if [ "$MODEL_FAILED" = false ]; then
        SUCCESS+=("${model}")
    else
        FAILED+=("${model}")
        echo -e "${YELLOW}Continuing with remaining models...${NC}"
    fi

    # Add delay between models to avoid rate limiting
    if [ $CURRENT -lt $TOTAL ]; then
        echo ""
        echo -e "${BLUE}Waiting 10 seconds before next model...${NC}"
        sleep 10
    fi
done

# Print summary
echo ""
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}PI EXPERIMENT SUMMARY${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${GREEN}Successful: ${#SUCCESS[@]}/${TOTAL}${NC}"
for model in "${SUCCESS[@]}"; do
    echo -e "  ${GREEN}✓${NC} ${model}"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed: ${#FAILED[@]}/${TOTAL}${NC}"
    for model in "${FAILED[@]}"; do
        echo -e "  ${RED}✗${NC} ${model}"
    done
fi

echo ""
echo -e "${BLUE}Results saved to: ${PROJECT_ROOT}/data/raw_experiments_pi/${NC}"
echo -e "${BLUE}=====================================================================${NC}"

# Exit with error if any failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi

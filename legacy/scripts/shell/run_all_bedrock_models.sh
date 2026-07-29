#!/bin/bash

# Run interference experiments for all AWS Bedrock models
# Usage: ./run_all_bedrock_models.sh [num_runs]
# Example: ./run_all_bedrock_models.sh 3  # Run each model 3 times

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
LEVELS="3 10 50 100 200 300 400 500"

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

    # Meta Llama 4 (1M context)
    "bedrock-llama-4-scout-17b"
    "bedrock-llama-4-maverick-17b"

    # Amazon Titan
    "bedrock-titan-large"
    "bedrock-titan-express"
    "bedrock-titan-lite"

    # Amazon Nova Gen1 (300K context)
    "bedrock-nova-micro"
    "bedrock-nova-lite"
    "bedrock-nova-pro"
    "bedrock-nova-premier"

    # Anthropic Claude 3 (200K context)
    "bedrock-claude-3-haiku"
    "bedrock-claude-3-sonnet"
    "bedrock-claude-3-opus"

    # Anthropic Claude 3.5 (200K context)
    "bedrock-claude-3.5-haiku"
    "bedrock-claude-3.5-sonnet"

    # Anthropic Claude 3.7 (200K context)
    "bedrock-claude-3.7-sonnet"

    # Anthropic Claude 4 (200K context)
    "bedrock-claude-4-sonnet"
    "bedrock-claude-4-opus"

    # Mistral (32K context)
    "bedrock-mistral-7b"
    "bedrock-mistral-8x7b"
    "bedrock-mistral-large"
    "bedrock-mistral-small"

    # Ministral (128K context)
    "bedrock-ministral-3b"
    "bedrock-ministral-8b"

    # DeepSeek R1 (64K context)
    "bedrock-deepseek-r1"

    # Qwen (256K context)
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
echo -e "${BLUE}Starting AWS Bedrock Model Interference Experiments${NC}"
echo -e "${BLUE}Testing ${TOTAL} models with levels: ${LEVELS}${NC}"
echo -e "${BLUE}Runs per model: ${NUM_RUNS}${NC}"
echo -e "${BLUE}Output directory: ${PROJECT_ROOT}/data/raw_experiments/${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# Create output directory
mkdir -p data/raw_experiments

# Check AWS credentials
echo -e "${YELLOW}Checking AWS credentials...${NC}"
if ! aws sts get-caller-identity &>/dev/null; then
    echo -e "${RED}✗ AWS credentials not configured or expired${NC}"
    echo -e "${RED}Please configure AWS credentials before running this script${NC}"
    exit 1
fi
echo -e "${GREEN}✓ AWS credentials valid${NC}"
echo ""

# Loop through each model
for model in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))

    echo ""
    echo -e "${GREEN}[${CURRENT}/${TOTAL}] Running experiment for: ${model}${NC}"
    echo -e "${GREEN}=====================================================================${NC}"

    # Run multiple times if requested
    MODEL_FAILED=false
    for run in $(seq 1 $NUM_RUNS); do
        if [ $NUM_RUNS -gt 1 ]; then
            echo -e "${YELLOW}Run ${run}/${NUM_RUNS}${NC}"
        fi

        # Run the experiment from project root
        if python scripts/core/run_interleaved_experiment.py \
            --model "${model}" \
            --levels ${LEVELS} \
            --output-dir data/raw_experiments; then
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
        echo -e "${BLUE}Waiting 15 seconds before next model...${NC}"
        sleep 15
    fi
done

# Print summary
echo ""
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}EXPERIMENT SUMMARY${NC}"
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
echo -e "${BLUE}Results saved to: ${PROJECT_ROOT}/data/raw_experiments/${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Export to Excel: python scripts/analysis/export_results_to_excel.py"
echo -e "  2. Calculate RIES:  python scripts/analysis/calculate_ries.py --input data/experiment_results.xlsx"

# Exit with error if any failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

PROJECT="/Users/jonathangadeaharder/projects/vidiomtm/german-spanish-english-eurobert-lemmatizer"
LOG_FILE="$PROJECT/artifacts/llm_validation/run_output.log"

cd "$PROJECT"
mkdir -p artifacts/llm_validation

# Source shell to get API key
source ~/.zshrc 2>/dev/null || true

echo "========================================" | tee "$LOG_FILE"
echo "Step 1: uv sync" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
uv sync 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Step 2: API key check" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
if [ -z "${SKAINET_EXTERNAL_API_KEY:-}" ] && [ -z "${SKAINET_API_KEY:-}" ] && [ -z "${API_KEY:-}" ]; then
    echo "ERROR: No API key found in env" | tee -a "$LOG_FILE"
    echo "Checking ~/.zshrc..." | tee -a "$LOG_FILE"
    grep -n 'SKAINET\|API_KEY' ~/.zshrc 2>/dev/null | tee -a "$LOG_FILE" || true
    # Try to export from zshrc
    export SKAINET_EXTERNAL_API_KEY="$(grep 'SKAINET_EXTERNAL_API_KEY' ~/.zshrc | head -1 | sed "s/.*=\"\([^\"]*\)\".*/\1/")"
    if [ -n "$SKAINET_EXTERNAL_API_KEY" ]; then
        echo "Found key in ~/.zshrc, exported." | tee -a "$LOG_FILE"
    else
        echo "FATAL: Cannot find API key. Set SKAINET_EXTERNAL_API_KEY manually." | tee -a "$LOG_FILE"
        exit 1
    fi
else
    echo "API key found: ${SKAINET_API_KEY:+set}${SKAINET_EXTERNAL_API_KEY:+set}${API_KEY:+set}" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Step 3: Test run (3 sentences/lang, dev files)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
uv run python scripts/llm_validate_gold.py --model external-glm52 --test 2>&1 | tee -a "$LOG_FILE"
TEST_EXIT=$?

if [ $TEST_EXIT -ne 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "Test failed (exit $TEST_EXIT). Trying external-tee..." | tee -a "$LOG_FILE"
    uv run python scripts/llm_validate_gold.py --model external-tee --test 2>&1 | tee -a "$LOG_FILE"
    TEST_EXIT=$?
fi

if [ $TEST_EXIT -ne 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "Test failed (exit $TEST_EXIT). Trying internal-direct..." | tee -a "$LOG_FILE"
    uv run python scripts/llm_validate_gold.py --model internal-direct --test 2>&1 | tee -a "$LOG_FILE"
    TEST_EXIT=$?
fi

if [ $TEST_EXIT -ne 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "FATAL: All test runs failed. Check script for bugs." | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Step 4: Full run (all 16 gold files)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Starting full validation at $(date)..." | tee -a "$LOG_FILE"

# Determine which model worked in test
MODEL="external-glm52"
if grep -q "DONE:" "$LOG_FILE"; then
    # Use the last model that produced DONE output
    MODEL=$(grep "model=" "$LOG_FILE" | tail -1 | sed 's/.*--model \([^ ]*\).*/\1/' 2>/dev/null || echo "external-glm52")
fi

echo "Using model: $MODEL" | tee -a "$LOG_FILE"
uv run python scripts/llm_validate_gold.py --model "$MODEL" --full 2>&1 | tee -a "$LOG_FILE"
FULL_EXIT=$?

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "COMPLETE: Full run exit=$FULL_EXIT" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Report: $PROJECT/artifacts/llm_validation/validation_${MODEL}_full.json" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

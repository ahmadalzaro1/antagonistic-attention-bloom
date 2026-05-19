#!/usr/bin/env bash
# Run ablation experiment for all 4 languages sequentially.
# Arabic resumes from checkpoint (12/72 done); others start fresh.
set -e
cd "$(dirname "$0")"

echo "========================================"
echo "  BLOOM-560m 288-Prompt Ablation Sweep  "
echo "  Started: $(date)"
echo "========================================"

echo ""
echo "[1/4] ARABIC (--resume, 12/72 checkpointed)"
python3 -u ablation_experiment.py configs/bloom_massive_arabic.json --resume
echo "Arabic done at $(date)"

echo ""
echo "[2/4] ENGLISH"
python3 -u ablation_experiment.py configs/bloom_massive_english.json
echo "English done at $(date)"

echo ""
echo "[3/4] FRENCH"
python3 -u ablation_experiment.py configs/bloom_massive_french.json
echo "French done at $(date)"

echo ""
echo "[4/4] CHINESE"
python3 -u ablation_experiment.py configs/bloom_massive_chinese.json
echo "Chinese done at $(date)"

echo ""
echo "========================================"
echo "  ALL 4 LANGUAGES COMPLETE"
echo "  Finished: $(date)"
echo "========================================"

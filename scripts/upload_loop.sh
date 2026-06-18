set -u
L=$1
for a in $(seq 1 25); do
  echo "[$L] attempt $a"
  landed=$(uv run python -c "from huggingface_hub import HfApi; print(any('int8' in s.rfilename for s in HfApi().model_info('Jonandrop/eurobert-lemma-$L-210m').siblings))" 2>/dev/null)
  [ "$landed" = "True" ] && { echo "[$L] LANDED"; exit 0; }
  timeout 200 env HF_HUB_ENABLE_HF_TRANSFER=0 uv run python scripts/upload_once.py "$L" && { echo "[$L] DONE"; exit 0; }
  echo "[$L] attempt $a ended rc=$? (timeout/err) — resume next"
done
echo "[$L] EXHAUSTED"; exit 1

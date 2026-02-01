EVAL_NOTE="v3-distilling-r2"
# EVAL_LIMIT=1
MODEL_CONFIG="llm.Qwen3-Coder-30B-toolcalling"

MAX_ITER=100
NUM_WORKERS=10

DATASET="SWE-bench-Live/SWE-bench-Live"
DATASET_DIR="/shared_workspace/yanruo/data/Public/SWE-bench-Live"
SPLIT="full"


COMMAND="poetry run python evaluation/benchmarks/swe_bench/run_infer_live.py \
--llm-config $MODEL_CONFIG \
--max-iterations $MAX_ITER \
--eval-num-workers $NUM_WORKERS \
--eval-note $EVAL_NOTE \
--dataset $DATASET \
--split $SPLIT \
--dataset_dir $DATASET_DIR"


if [ -n "$EVAL_LIMIT" ]; then
echo "EVAL_LIMIT: $EVAL_LIMIT"
COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
fi

# Run the command
export DEBUG=false;
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true

eval $COMMAND


# EVAL_LIMIT=1
# EVAL_ONLY=true

# MODEL_CONFIG="llm.Qwen3-Coder-30B-toolcalling"
MODEL_CONFIG="llm.deepswe32b_cyber3"

MAX_ITER=50
NUM_WORKERS=10

DATASET="SWE-bench-Rebench"
DATASET_DIR="/shared_workspace/yanruo/data/Public/SWE-rebench"
SPLIT="test"

# DATASET="SWE-bench-Live/SWE-bench-Live"
# DATASET_DIR="/shared_workspace/yanruo/data/Public/SWE-bench-Live"
# SPLIT="full"

EVAL_NOTE="v3-distilling-r3"
# EVAL_NOTE="v3-distilling-r2"


COMMAND="poetry run python evaluation/benchmarks/swe_bench/run_infer_rebench.py \
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

if [ "$EVAL_ONLY" ]; then
echo "EVAL_ONLY: $EVAL_ONLY"
COMMAND="$COMMAND --eval_only"
fi


# Run the command
export DEBUG=false;
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true
echo $COMMAND
eval $COMMAND

#!/bin/bash
# setup.sh — re-run after every spot instance restart
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

pip install torch transformers peft vllm trl datasets \
    scikit-learn joblib openai tiktoken pandas tqdm matplotlib

export PYTHONPATH="$REPO_DIR:$REPO_DIR/arlsat"
grep -qxF "export PYTHONPATH=$REPO_DIR:$REPO_DIR/arlsat" ~/.bashrc \
    || echo "export PYTHONPATH=$REPO_DIR:$REPO_DIR/arlsat" >> ~/.bashrc

python -c "from big_math.trace.load_data import load_data; from icl.gradient.analysis import GradientAnalyzer; print('imports ok')"

echo ""
echo "Setup complete. Next: cd $REPO_DIR/big_math && python smoke_test.py"

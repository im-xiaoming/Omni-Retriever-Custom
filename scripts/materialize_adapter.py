#!/usr/bin/env python3
"""pt adapter + LoRA-only checkpoint (``<run>/best``, retrieval_eval.save_lora) -> a full adapter directory that
PeftModel.from_pretrained / omniretriever.cli load. The heads (classify_linear, beats_ln, beats_proj) are the pt ones.

    python scripts/materialize_adapter.py <pt adapter dir> <run>/best <out dir>
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
from qwenvl.train.retrieval_eval import materialize_adapter  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    print(materialize_adapter(*sys.argv[1:]))

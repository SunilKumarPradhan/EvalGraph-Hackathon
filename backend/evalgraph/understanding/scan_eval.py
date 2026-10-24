"""Does the repo ship an evaluation dataset?"""

from __future__ import annotations

import re

from .buffer import RepoBuffer

EVAL_PATH = re.compile(r"(^|/)(evals?|evaluation|benchmarks?)/.*\.(jsonl|json|csv|ya?ml)$|(^|/)golden[^/]*\.(jsonl|json|csv)$"
                       r"|(^|/)(test|eval)_?(set|dataset|cases)[^/]*\.(jsonl|json|csv)$")


def has_eval_suite(buffer: RepoBuffer) -> bool:
    return any(EVAL_PATH.search(p) for p in [*buffer.files, *buffer.other_paths])

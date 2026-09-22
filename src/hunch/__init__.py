"""hunch — plain function verbs on Jev. Lists in, lists out. Jev decides; an LLM may propose."""

from hunch.answer import Answer, Feeling, MultiAnswer, Pick, Ranked, Rating
from hunch.client import Client, configure, default
from hunch.engine import Plan, dry_run
from hunch.evaluate import Evaluation, Threshold, evaluate, tune_threshold
from hunch.exceptions import HunchError
from hunch.generate import generate, generate_async
from hunch.llm import LanguageModel, OpenAICompat, cerebras, openai, openrouter
from hunch.shapes import Shape, ShapePolicy
from hunch.usage import Usage
from hunch.verbs import (
    Check,
    Classify,
    Rate,
    ask,
    ask_async,
    check,
    check_async,
    classify,
    classify_async,
    pick,
    pick_async,
    rank,
    rank_async,
    score,
    score_async,
    where,
    where_async,
)

try:  # df.hunch.<verb>() when pandas is installed
    import hunch.accessor as _accessor  # noqa: F401
except ImportError:
    pass

__all__ = [
    "Answer",
    "Check",
    "Classify",
    "Client",
    "Evaluation",
    "Plan",
    "Threshold",
    "Feeling",
    "HunchError",
    "LanguageModel",
    "MultiAnswer",
    "OpenAICompat",
    "Pick",
    "Ranked",
    "Rate",
    "Rating",
    "Shape",
    "ShapePolicy",
    "Usage",
    "ask",
    "ask_async",
    "cerebras",
    "check",
    "check_async",
    "classify",
    "classify_async",
    "configure",
    "default",
    "dry_run",
    "evaluate",
    "generate",
    "generate_async",
    "openai",
    "openrouter",
    "pick",
    "pick_async",
    "rank",
    "rank_async",
    "score",
    "score_async",
    "tune_threshold",
    "where",
    "where_async",
]

__version__ = "0.7.0"

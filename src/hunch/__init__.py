"""hunch — plain function verbs on Jev. Lists in, lists out. Jev decides; an LLM may propose."""

from hunch.answer import Answer, Feeling, MultiAnswer, Pick, Ranked, Rating
from hunch.client import Client, configure, default
from hunch.combine import Refined, discover, discover_async, refine, refine_async
from hunch.combine import verify as verify
from hunch.engine import Plan, dry_run
from hunch.evaluate import Evaluation, Threshold, evaluate, tune_threshold
from hunch.exceptions import HunchError
from hunch.extract import PATTERNS, extract, extract_async
from hunch.generate import generate, generate_async
from hunch.llm import LanguageModel, OpenAICompat, anthropic, azure, cerebras, ollama, openai, openrouter
from hunch.shapes import Shape, ShapePolicy
from hunch.usage import Usage
from hunch.verbs import (
    Check,
    Classify,
    Rate,
    Tree,
    ask,
    ask_async,
    check,
    check_async,
    classify,
    classify_async,
    pick,
    pairs,
    pick_async,
    rank,
    rank_async,
    score,
    score_async,
    verify_async,
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
    "Tree",
    "Feeling",
    "HunchError",
    "LanguageModel",
    "MultiAnswer",
    "OpenAICompat",
    "PATTERNS",
    "Pick",
    "Ranked",
    "Rate",
    "Rating",
    "Refined",
    "Shape",
    "ShapePolicy",
    "Usage",
    "anthropic",
    "ask",
    "azure",
    "ask_async",
    "cerebras",
    "check",
    "check_async",
    "classify",
    "classify_async",
    "configure",
    "default",
    "discover",
    "discover_async",
    "dry_run",
    "evaluate",
    "extract",
    "extract_async",
    "generate",
    "generate_async",
    "ollama",
    "openai",
    "openrouter",
    "pairs",
    "pick",
    "pick_async",
    "rank",
    "rank_async",
    "refine",
    "refine_async",
    "score",
    "score_async",
    "tune_threshold",
    "verify",
    "verify_async",
    "where",
    "where_async",
]

__version__ = "0.9.0"

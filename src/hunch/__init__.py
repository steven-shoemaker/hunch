"""hunch — plain function verbs on Jev. Lists in, lists out. Jev decides; an LLM may propose."""

from hunch.answer import Answer, Feeling, MultiAnswer, Pick, Ranked, Rating
from hunch.client import Client, configure, default
from hunch.exceptions import HunchError
from hunch.generate import generate, generate_async
from hunch.llm import LanguageModel, OpenAICompat, cerebras, openai, openrouter
from hunch.shapes import Shape, ShapePolicy
from hunch.usage import Usage
from hunch.verbs import (
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
)

__all__ = [
    "Answer",
    "Client",
    "Feeling",
    "HunchError",
    "LanguageModel",
    "MultiAnswer",
    "OpenAICompat",
    "Pick",
    "Ranked",
    "Rating",
    "Shape",
    "ShapePolicy",
    "Usage",
    "cerebras",
    "check",
    "check_async",
    "classify",
    "classify_async",
    "configure",
    "default",
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
]

__version__ = "0.2.1"

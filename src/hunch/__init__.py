"""hunch — Jev as a column primitive, with optional LLM roles that only propose."""

from hunch.answer import Answer, Feeling, Rating
from hunch.client import Hunch, connect
from hunch.exceptions import HunchError, NoSessionError
from hunch.llm import LanguageModel, OpenAICompat, cerebras, openai, openrouter
from hunch.over import OverJob, over
from hunch.role import Draft, Role, draft, role
from hunch.session import ask, rate
from hunch.shapes import Shape, ShapePolicy
from hunch.subject import Subject
from hunch.usage import Usage

__all__ = [
    "Answer",
    "Draft",
    "Feeling",
    "Hunch",
    "HunchError",
    "LanguageModel",
    "NoSessionError",
    "OpenAICompat",
    "OverJob",
    "Rating",
    "Role",
    "Shape",
    "ShapePolicy",
    "Subject",
    "Usage",
    "ask",
    "cerebras",
    "connect",
    "draft",
    "openai",
    "openrouter",
    "over",
    "rate",
    "role",
]

__version__ = "0.1.0"

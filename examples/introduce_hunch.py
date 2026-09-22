"""Draft 20 tweets with an LLM, rank them with Jev, let Jev pick the winner, then polish it.

Needs TYPESAFE_API_KEY and OPENROUTER_API_KEY in the environment (e.g. `set -a; . .env`).
"""

from pathlib import Path

import hunch

readme = (Path(__file__).parents[1] / "README.md").read_text()
hunch.configure(llm=hunch.openrouter(), cache="~/.cache/hunch")

tweets = hunch.generate(
    str,
    n=20,
    instructions="Tweets under 280 characters introducing hunch to Python developers. "
    "Lead with a pain, show a snippet or the install line, no hashtags, no hype words.",
    context={"readme": readme},
)

ranked = hunch.rank(
    tweets,
    {
        "hook": "Does the first line name a real developer pain?",
        "proof": "Does it show code or an install line rather than just claim?",
        "specific": "How concrete is it, versus generic 'AI-powered' language?",
    },
    levels=["weak", "okay", "strong", "excellent"],
    weights={"hook": 2, "proof": 1, "specific": 1},
)
for tweet, composite in ranked:
    print(f"{composite:.2f}  {tweet}\n")

winner = hunch.pick(
    [tweet for tweet, _ in ranked[:5]],
    "the tweet most likely to make a Python developer install hunch",
)
print("WINNER\n", winner)

final = hunch.refine(
    winner,
    {
        "accurate": "describes the library correctly according to the readme in context",
        "install": "includes the command pip install hunch-jev",
        "calm": "uses no exclamation marks, hashtags, or hype words",
        "short": "is under 280 characters",
    },
    context={"readme": readme},
    detail=True,
)
print(f"\nFINAL (passed={final.passed}, rewrites={final.rounds})\n", final.text)

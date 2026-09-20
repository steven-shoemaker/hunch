"""Classify job titles with an Enum, then route on how sure Jev was."""

from enum import Enum

import pandas as pd

import hunch


class Function(Enum):
    SALES = "Sales"
    ENGINEERING = "Engineering"
    MARKETING = "Marketing"
    OPERATIONS = "Operations"
    OTHER = "Other"


class Level(Enum):
    IC = "Individual Contributor"
    MANAGER = "Manager"
    DIRECTOR = "Director"
    EXEC = "Executive"


prospects = pd.DataFrame(
    {
        "title": [
            "VP Sales, EMEA",
            "Staff Software Engineer",
            "Head of Growth",
            "Account Manager",
            "Chief of Staff",
            "Senior Manager, Demand Gen",
            "Program Manager",
            "VP Sales, EMEA",  # duplicate: asked once, joined twice
        ]
    }
)

# Enum in, Enum members out. A Series keeps its index.
prospects["function"] = hunch.classify(prospects["title"], Function, instructions="Which business function does this title belong to?")

# detail=True gives the whole distribution; .on() turns shape into policy.
levels = hunch.classify(
    prospects["title"],
    Level,
    instructions="What level of rank does this title state? 'Manager' means people management, not 'Account Manager'.",
    detail=True,
)


def route(title: str, answer: hunch.Answer) -> str:
    return answer.on(
        sure=answer.label.value,
        # two levels are close: ask Jev to pick between just those two
        split=lambda: hunch.classify(title, answer.top2, instructions="Which of these two fits the title better?"),
        unsure="review",
    )


prospects["level"] = [route(t, a) for t, a in zip(prospects["title"], levels)]
prospects["level_shape"] = levels.map(lambda a: a.shape)
prospects["level_p"] = levels.map(lambda a: round(a.p, 2))

print(prospects.assign(function=prospects["function"].map(lambda f: f.value)).to_string(index=False))
u = hunch.default().usage
print(f"\n{u.calls} Jev calls for {len(prospects)} rows ({u.hits} cache hits).")

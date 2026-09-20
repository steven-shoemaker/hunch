"""Generate dating profiles as typed data, screen them with Jev, then run semantic WHERE queries.

Needs TYPESAFE_API_KEY and OPENROUTER_API_KEY in the environment (or pass them to configure()).
"""

from dataclasses import dataclass

import pandas as pd

import hunch
from hunch import Check, Classify, ask

hunch.configure(llm=hunch.openrouter())


@dataclass
class Profile:
    name: str
    age: int
    bio: str
    first_date_idea: str


profiles = hunch.generate(Profile, n=15, instructions=(
    "Dating app profiles for people aged 28 to 40 in Denver. Bios are 2 to 3 sentences, "
    "specific, and varied: some charming, some boring, a few with subtle red flags."))

ME = "Loves early mornings, trail running, and quiet weekends. Allergic to cats. Does not drink. Wants kids."

df = pd.DataFrame(profiles)

# Two questions about every row in one request each. Label descriptions keep "vibe" honest.
df = df.join(ask(df, {
    "vibe": Classify({
        "outdoorsy": "hobbies are mostly outside: hiking, skiing, fishing, running",
        "homebody": "prefers evenings in, cooking, TV, quiet weekends",
        "nightlife": "bars, shows, going out is the default weekend",
        "workaholic": "the job dominates the bio and the schedule",
        "unclear": "the bio doesn't say enough to tell",
    }),
    "red_flag": Check("the bio contains something a careful reader would consider a red flag"),
}))

# Compatibility is judged against ME, so context goes on this call only.
df = df.join(hunch.score(df, ["clear mismatch", "some friction", "plausible", "strong match"],
                         instructions="How compatible is this profile with the person described in context?",
                         context={"looking_for": ME}, detail=True))

best = hunch.pick(df[~df.red_flag], "the best first date for the person described in context", context={"looking_for": ME})
df["message"] = df.index == best

# Semantic WHERE. Evidence questions filter well; predictions are better ranked than thresholded.
cats = df.hunch.where("probably likes cats", columns=["name", "age", "bio"], threshold=0.7)
dogs = df.hunch.where("probably likes dogs", columns=["name", "age", "bio"])
scary = df.hunch.where(
    "might yell at a waiter for getting their order wrong",
    columns=["name", "age", "bio"],
    detail=True,
).sort_values("match_p", ascending=False).head(5)

pd.set_option("display.width", 200, "display.max_colwidth", 60)
print(df.sort_values("score", ascending=False)[["name", "age", "vibe", "red_flag", "score_level", "score_shape", "message"]].to_string())
print(f"\nMessage {df.loc[best, 'name']}: \"{df.loc[best, 'first_date_idea']}\"")
print("\nCat people:", ", ".join(cats["name"]) or "none")
print("Dog people:", ", ".join(dogs["name"]) or "none")
print("\nMost likely to yell at a waiter:")
print(scary[["name", "match_p"]].to_string(index=False))

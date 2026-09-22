"""Find the themes in a pile of reviews nobody has categorized, then file every review.

An LLM generates fake reviews and proposes the themes. Jev files every review, and only
the reviews it's unsure about go back to the LLM. Then a semantic WHERE finds churn risk.

Needs TYPESAFE_API_KEY and OPENROUTER_API_KEY in the environment (or pass keys to configure()).
"""

import pandas as pd

import hunch

hunch.configure(llm=hunch.openrouter())

reviews = pd.Series(hunch.generate(str, n=60, instructions=(
    "Short, realistic customer reviews for a meal-kit delivery company. One to three sentences each. "
    "Mix praise and complaints about delivery, food quality, recipes, pricing, and the app.")))

themes = reviews.hunch.discover(5, instructions="by what the customer is talking about")
for name, description in themes.items():
    print(f"{name:<28} {description}")

tagged = reviews.hunch.classify(themes, split="rematch", unsure=hunch.default().llm, detail=True)
churn = reviews.hunch.where("the customer is likely to cancel soon", detail=True)

df = pd.DataFrame({"review": reviews}).join(tagged[["label", "label_shape", "label_by"]]).join(churn[["match_p"]])
pd.set_option("display.width", 200, "display.max_colwidth", 80)
print("\n", df["label"].value_counts().to_string())
print(f"\nEscalated to the LLM: {(df['label_by'] == 'llm').sum()} of {len(df)}")
print("\nMost likely to cancel:")
print(df.nlargest(5, "match_p")[["label", "match_p", "review"]].to_string(index=False))

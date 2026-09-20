"""Find angry customer reviews in a column. check() over a Series is a boolean mask."""

import pandas as pd

import hunch

reviews = pd.DataFrame(
    {
        "id": range(1, 11),
        "review": [
            "Third time the package arrived crushed. Nobody answers support. Done with this company.",
            "Works as described. Setup took five minutes.",
            "I asked for a refund two weeks ago and keep getting form emails. This is a scam.",
            "Decent for the price. Battery could be better.",
            "Absolutely furious. Charged twice and the second charge is still there.",
            "Love it. Bought one for my sister too.",
            "Not what I expected but the return was painless.",
            "Your app deleted my playlists and your 'help' bot told me to restart. Unbelievable.",
            "Shipping was slow but the product is fine.",
            "Meh.",
        ],
    }
)

# One line: a Series in, a boolean Series with the same index out.
angry = reviews[hunch.check(reviews["review"], "is an angry customer review")]
print(f"{len(angry)} angry reviews:")
for row in angry.itertuples():
    print(f"  #{row.id}  {row.review[:70]}")

# The probability is there when you want to rank instead of filter, or tune the cutoff.
feelings = hunch.check(reviews["review"], "is an angry customer review", detail=True)
ranked = reviews.assign(p_angry=feelings.map(lambda f: f.p)).sort_values("p_angry", ascending=False)
print("\nMost to least angry:")
print(ranked[["id", "p_angry"]].to_string(index=False))

# Several checks about the same column go in one request per row.
flags = hunch.check(
    reviews["review"],
    {"angry": "is an angry customer review", "refund": "asks for or mentions a refund", "churn": "says they will stop being a customer"},
)
print("\nFlags for the first three rows:")
for row_id, flag in zip(reviews["id"].head(3), flags.head(3)):
    print(f"  #{row_id}  {flag}")

u = hunch.default().usage
print(f"\n{u.calls} Jev calls, {u.hits} cache hits.")

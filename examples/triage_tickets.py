"""Triage support tickets: several scores in one request, plus multi-label tags."""

import hunch

tickets = [
    "Checkout page returns 500 for every customer since 9am. Revenue is zero right now.",
    "The export button is misaligned by a few pixels on Safari.",
    "I can't log in on mobile. Desktop works. I've tried resetting my password twice.",
    "Feature request: dark mode for the dashboard.",
    "Invoice #4411 charged us twice. Please refund one and confirm in writing.",
]

SEVERITY = [
    "cosmetic: visual or wording issue, nothing blocked",
    "degraded: something is broken but there is a workaround",
    "blocked: a user cannot complete a core task",
    "outage: many users or revenue affected right now",
]

# Two ordered scales, scored in one request per ticket. instructions as a dict -> a dict per ticket.
scores = hunch.score(
    tickets,
    SEVERITY,
    instructions={
        "severity": "How severe is the problem described?",
        "urgency": "How urgent is a response, judging from what the customer says about timing and impact?",
    },
    detail=True,
)

# Multi-label: one Noul per tag, thresholded in code.
tags = hunch.classify(
    tickets,
    {
        "bug": "something that used to work is broken",
        "billing": "charges, invoices, refunds",
        "auth": "login, passwords, sessions",
        "feature-request": "asks for something new",
        "ui": "layout, styling, copy",
    },
    multi_label=True,
    threshold=0.6,
)

for text, s, t in zip(tickets, scores, tags):
    sev, urg = s["severity"], s["urgency"]
    print(f"[{sev.level.split(':')[0]:>8} / urgency {urg.score:.1f}] {', '.join(t) or '-'}")
    print(f"    {text}")

# Policy stays in code: page someone only when both scales agree it's bad.
page = [t for t, s in zip(tickets, scores) if s["severity"].score >= 2.5 and s["urgency"].score >= 2.5]
print(f"\nPage on-call for {len(page)} ticket(s).")
u = hunch.default().usage
print(f"{u.calls} Jev calls, {u.hits} cache hits.")

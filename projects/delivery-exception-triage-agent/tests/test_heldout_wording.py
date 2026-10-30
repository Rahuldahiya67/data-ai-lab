"""Customer wording the generator never produces.

The sample dataset is built from a fixed set of message templates, so a score on
it says little about wording the planner has not seen. These messages were
written separately and are not used to tune ``detect_intent``. The known misses
are kept as strict expected failures: if a change fixes one, the test suite says
so and the README figure has to be updated.
"""

import pytest

from delivery_exception_triage_agent.planner import detect_intent

HELD_OUT = [
    ("The glass bottle inside was shattered when I opened the box.", "damaged"),
    ("Box came in dented and the lamp does not work.", "damaged"),
    ("The bag was torn open and half the items are missing.", "damaged"),
    ("Item arrived but it's broke.", "damaged"),
    ("My order is leaking oil all over the floor.", "damaged"),
    ("Product came damgaed, need a refund.", "damaged"),
    ("No damage to the box, but it is two days late. Where is it?", "status"),
    ("Package was not damaged, I just want to know when it arrives.", "status"),
    ("The app shows delivered yet I never got anything.", "not_received"),
    ("I didnt receive the parcel, the status is wrong.", "not_received"),
    ("Thank you, the parcel reached safely.", "thanks"),
    ("Received it, thnaks a lot!", "thanks"),
    ("When will my order arrive?", "status"),
    ("Still waiting for my shipment, no update since Monday.", "status"),
    ("Is my parcel lost? It has been a week.", "status"),
    ("Driver could not find the house, please try again tomorrow.", "status"),
    ("The courier never came.", "status"),
]

KNOWN_MISSES = [
    # "have not received" and "not here" are not in the not-received patterns.
    ("Marked as delivered but I have not received it.", "not_received"),
    ("Says delivered. It is not here.", "not_received"),
    # Praise without a thank-you word.
    ("Parcel came on time, great service.", "thanks"),
]


@pytest.mark.parametrize("message, intent", HELD_OUT)
def test_held_out_wording(message, intent):
    assert detect_intent(message) == intent


@pytest.mark.xfail(strict=True, reason="known baseline miss on held-out wording")
@pytest.mark.parametrize("message, intent", KNOWN_MISSES)
def test_known_misses_on_held_out_wording(message, intent):
    assert detect_intent(message) == intent


def test_held_out_score_matches_the_readme():
    cases = HELD_OUT + KNOWN_MISSES
    correct = sum(detect_intent(m) == i for m, i in cases)
    assert (correct, len(cases)) == (17, 20)

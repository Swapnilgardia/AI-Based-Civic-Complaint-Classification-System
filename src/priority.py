"""
priority.py
-----------
Rule-based priority assignment for classified civic complaints.

The abstract calls for assigning "priority levels based on the identified
complaint category" using predefined rules (not a learned model) - this
mirrors how many real municipal grievance systems work, since priority
often depends on public-safety/health impact rather than something you'd
want a black-box classifier deciding.

Rules (editable - tune these to match your local municipal SOP):
    Water Leakage      -> High     (wastage of potable water, possible flooding)
    Road Damage        -> High     (accident risk)
    Drainage Blockage  -> Medium   (health hazard, worsens in monsoon)
    Garbage             -> Medium   (health/hygiene hazard)
    Streetlight Fault  -> Low      (safety concern but not immediately hazardous)

Confidence also nudges priority: a very low-confidence prediction is flagged
for manual review regardless of category, since the auto-classification may
be unreliable.
"""

BASE_PRIORITY = {
    "Water Leakage": "High",
    "Road Damage": "High",
    "Drainage Blockage": "Medium",
    "Garbage": "Medium",
    "Streetlight Fault": "Low",
}

LOW_CONFIDENCE_THRESHOLD = 0.55


def assign_priority(category: str, confidence: float) -> dict:
    """Returns a dict with the assigned priority level and a human-readable reason."""
    base = BASE_PRIORITY.get(category, "Medium")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return {
            "priority": base,
            "flagged_for_review": True,
            "reason": (
                f"Category predicted as '{category}' with low confidence "
                f"({confidence:.0%}). Default priority '{base}' assigned, "
                "but this complaint is flagged for manual verification."
            ),
        }

    return {
        "priority": base,
        "flagged_for_review": False,
        "reason": f"Category '{category}' predicted with {confidence:.0%} confidence -> priority '{base}'.",
    }

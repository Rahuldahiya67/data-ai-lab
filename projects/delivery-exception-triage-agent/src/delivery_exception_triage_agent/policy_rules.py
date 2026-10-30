"""Policy thresholds shared by the generator (ground truth) and the guardrails.

Keeping them in one module means the labels and the checks cannot drift apart.
"""

# Refunds up to this declared value can be issued without a supervisor.
REFUND_AUTO_LIMIT_INR = 5000.0

# A shipment in transit with no scan for this long is treated as stuck (lost).
STUCK_AFTER_HOURS = 72

# A redelivery must be booked between tomorrow and this many days ahead.
REDELIVERY_MAX_DAYS_AHEAD = 7

# Escalation reasons the desk recognises.
ESCALATION_REASONS = (
    "refund_above_limit",
    "delivery_dispute",
    "shipment_not_found",
    "policy_block",
    "unclear_request",
    "step_budget_exhausted",
    "planner_error",
)

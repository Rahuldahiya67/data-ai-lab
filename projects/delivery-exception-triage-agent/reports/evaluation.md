# Triage evaluation: rule-based

300 tickets, one shared action ledger, tickets in file order.

| Metric | Value |
|---|---|
| Action accuracy | 100.0% |
| Resolution accuracy (action + amount/date/reason) | 100.0% |
| Tool-call precision | 100.0% |
| Tool-call recall | 100.0% |
| Tool errors | 0 |
| Policy blocks | 0 |
| Planner errors | 0 |
| Step budget exhausted | 0 |

## By scenario

| Scenario | Correct | Tickets | Accuracy |
|---|---:|---:|---:|
| damaged_high_value | 27 | 27 | 100% |
| damaged_low_value | 33 | 33 | 100% |
| delivered_not_received | 33 | 33 | 100% |
| delivered_thanks | 9 | 9 | 100% |
| failed_bad_address | 26 | 26 | 100% |
| failed_customer_absent | 26 | 26 | 100% |
| late_moving | 60 | 60 | 100% |
| late_stuck | 28 | 28 | 100% |
| on_time_inquiry | 42 | 42 | 100% |
| unknown_shipment | 16 | 16 | 100% |

## Wrong actions

None.

## Mistakes (0)

None.

# Triage evaluation: rule-based

300 tickets, one shared action ledger, tickets in file order.

| Metric | Value |
|---|---|
| Action accuracy | 97.7% |
| Resolution accuracy (action + amount/date/reason) | 95.3% |
| Tool-call precision | 99.2% |
| Tool-call recall | 99.0% |
| Tool errors | 0 |
| Policy blocks | 0 |
| Planner errors | 0 |
| Step budget exhausted | 0 |

## By scenario

| Scenario | Correct | Tickets | Accuracy |
|---|---:|---:|---:|
| damaged_high_value | 22 | 27 | 81% |
| damaged_low_value | 27 | 33 | 82% |
| delivered_not_received | 31 | 33 | 94% |
| delivered_thanks | 8 | 9 | 89% |
| failed_bad_address | 26 | 26 | 100% |
| failed_customer_absent | 26 | 26 | 100% |
| late_moving | 60 | 60 | 100% |
| late_stuck | 28 | 28 | 100% |
| on_time_inquiry | 42 | 42 | 100% |
| unknown_shipment | 16 | 16 | 100% |

## Wrong actions

| Expected | Actual | Count |
|---|---|---:|
| close_ticket | escalate | 1 |
| issue_refund | escalate | 6 |

## Mistakes (14)

| Ticket | Scenario | Message | Expected | Got |
|---|---|---|---|---|
| TKT-00013 | damaged_low_value | Received SHP-00013 smashed, totally unusable. | issue_refund (842.16 INR) | escalate (unclear_request) |
| TKT-00099 | damaged_low_value | Received SHP-00099 smashed, totally unusable. | issue_refund (3,979.98 INR) | escalate (unclear_request) |
| TKT-00108 | damaged_low_value | Recieved SHP-00108 smashed, totally unusable. | issue_refund (1,249.57 INR) | escalate (unclear_request) |
| TKT-00115 | delivered_not_received | Tracking says SHP-00115 was delivered but I never erceived it. | escalate (delivery_dispute) | escalate (unclear_request) |
| TKT-00133 | damaged_high_value | Received SHP-00133 smashed, totally unusable. | escalate (refund_above_limit) | escalate (unclear_request) |
| TKT-00151 | damaged_high_value | Received SHP-00151 smashed, totally unusable. | escalate (refund_above_limit) | escalate (unclear_request) |
| TKT-00202 | damaged_high_value | Received SHP-00202 smashed, totally unusable. | escalate (refund_above_limit) | escalate (unclear_request) |
| TKT-00215 | delivered_thanks | Got SHP-00215, htanks for the quick delivery. | close_ticket | escalate (unclear_request) |
| TKT-00218 | damaged_high_value | Received SHP-00218 smashed, totally unusable. | escalate (refund_above_limit) | escalate (unclear_request) |
| TKT-00220 | delivered_not_received | Nobody here recevied SHP-00220. Who signed for it? | escalate (delivery_dispute) | escalate (unclear_request) |
| TKT-00239 | damaged_high_value | Received SHP-00239 smashed, totally unusable. | escalate (refund_above_limit) | escalate (unclear_request) |
| TKT-00249 | damaged_low_value | Received SHP-00249 smashed, totally unusable. | issue_refund (3,957.09 INR) | escalate (unclear_request) |
| TKT-00260 | damaged_low_value | Received SHP-00260 smashed, totally unusable. | issue_refund (1,795.85 INR) | escalate (unclear_request) |
| TKT-00267 | damaged_low_value | Received SHP-00267 smashed, totally unusable. | issue_refund (4,985.72 INR) | escalate (unclear_request) |

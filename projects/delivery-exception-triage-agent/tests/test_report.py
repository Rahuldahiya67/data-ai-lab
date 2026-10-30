from delivery_exception_triage_agent.evaluate import evaluate
from delivery_exception_triage_agent.models import LabelledTicket, Resolution, Ticket
from delivery_exception_triage_agent.planner import RuleBasedPlanner
from delivery_exception_triage_agent.report import to_json, to_markdown


def test_markdown_report_escapes_pipes_in_messages(in_transit_shipment, as_of):
    t = Ticket("TKT-00001", "SHP-00001", as_of, "thanks | all good\nbye")
    labelled = LabelledTicket(t, "delivered_thanks", Resolution("close_ticket"))
    report = evaluate({"SHP-00001": in_transit_shipment}, [labelled], RuleBasedPlanner())
    md = to_markdown(report, "rule-based")
    assert "thanks \\| all good bye" in md
    assert "| close_ticket | send_eta_update | 1 |" in md
    data = to_json(report, "rule-based")
    assert data["mistakes"][0]["expected"] == "close_ticket"
    assert data["summary"]["resolution_accuracy"] == 0.0

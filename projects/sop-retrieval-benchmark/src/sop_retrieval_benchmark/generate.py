"""Synthetic SOP corpus and labelled question set.

Every topic exists once per site, and each site's version has its own numbers.
That is what makes the benchmark hard in a realistic way: the Delhi and Mumbai
returns SOPs share almost all of their wording, so a retriever that ignores the
site in the question will often return the right rule for the wrong site.

Each fact is generated together with the question that asks for it, so every
question has a known gold document, gold section and expected answer.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Site:
    code: str
    city: str


SITES: tuple[Site, ...] = (
    Site("DEL-01", "Delhi"),
    Site("BOM-02", "Mumbai"),
    Site("BLR-03", "Bengaluru"),
    Site("MAA-04", "Chennai"),
    Site("CCU-05", "Kolkata"),
)


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    topic: str
    site_code: str
    sections: list[Section] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Question:
    qid: str
    question: str
    gold_doc_id: str
    gold_section: str
    answer: str
    topic: str
    site_code: str
    # "lexical": shares key terms with the SOP text.
    # "paraphrase": uses different words for the key term (e.g. "refrigerated
    # trailer" for "reefer"), which pure keyword search tends to miss.
    style: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FactTemplate:
    heading: str
    sentence: str  # SOP wording, formatted with site fields and params
    lexical_q: str
    paraphrase_q: str
    answer: str
    params: dict  # name -> tuple of candidate values


# Background sentences that every site's version of a section shares. They make
# the sections read like real SOP text and, deliberately, make site versions
# look alike to a keyword retriever.
_FILLER = {
    "scope": "This procedure applies to all shifts, contract staff and visiting carrier drivers. "
    "Deviations must be approved in writing by the site operations manager.",
    "records": "Every step is logged in the warehouse management system with the user ID and timestamp. "
    "Paper records are scanned and kept for three years.",
}


TOPICS: dict[str, dict] = {
    "cold-chain": {
        "title": "Cold Chain Handling",
        "facts": [
            FactTemplate(
                "Temperature range",
                "Chilled pharmaceutical pallets at {city} ({code}) must be held between {lo} and {hi} "
                "degrees Celsius from receipt until dispatch.",
                "What temperature range must chilled pharmaceutical pallets be held at in {ref}?",
                "How cold do we keep chilled medicine pallets at {ref}?",
                "{lo} to {hi} degrees Celsius",
                {"lo": (2, 3), "hi": (6, 8)},
            ),
            FactTemplate(
                "Excursion response",
                "If a data logger records a temperature excursion longer than {mins} minutes, the shift lead "
                "quarantines the pallet and raises a quality hold.",
                "How long can a temperature excursion last before the pallet is quarantined at {ref}?",
                "After how many minutes out of range does a shipment get put on hold at {ref}?",
                "{mins} minutes",
                {"mins": (15, 20, 30, 45)},
            ),
            FactTemplate(
                "Reefer pre-cooling",
                "Reefer trailers must be pre-cooled for at least {pc} minutes before loading starts.",
                "How long must reefer trailers be pre-cooled before loading at {ref}?",
                "How much time should a refrigerated truck run its cooling before we put stock on it at {ref}?",
                "{pc} minutes",
                {"pc": (20, 30, 40, 60)},
            ),
        ],
    },
    "returns": {
        "title": "Customer Returns Processing",
        "facts": [
            FactTemplate(
                "Return window",
                "Customer returns are accepted at {city} ({code}) up to {days} days after delivery when the "
                "original packing slip is present.",
                "How many days after delivery are customer returns accepted at {ref}?",
                "What is the deadline for a buyer to send goods back at {ref}?",
                "{days} days",
                {"days": (7, 10, 14, 30)},
            ),
            FactTemplate(
                "Grading",
                "Returned units are graded A, B or C within {hrs} hours of arrival, and grade C units are moved "
                "to the liquidation cage.",
                "Within how many hours are returned units graded at {ref}?",
                "How quickly do we have to inspect and classify sent-back items at {ref}?",
                "{hrs} hours",
                {"hrs": (24, 48, 72)},
            ),
            FactTemplate(
                "Refund release",
                "Refunds are released once the returned item is scanned into returns bay {bay}.",
                "Which returns bay must a returned item be scanned into before the refund is released at {ref}?",
                "Where does a sent-back item need to be booked in before the customer gets money back at {ref}?",
                "bay {bay}",
                {"bay": ("R1", "R2", "R3", "R4")},
            ),
        ],
    },
    "hazmat": {
        "title": "Hazardous Materials Storage",
        "facts": [
            FactTemplate(
                "Flammable liquid limits",
                "Class 3 flammable liquids are limited to {qty} litres per rack bay at {city} ({code}).",
                "What is the limit for class 3 flammable liquids per rack bay at {ref}?",
                "How much petrol-type product can sit in one storage position at {ref}?",
                "{qty} litres",
                {"qty": (200, 250, 400, 500)},
            ),
            FactTemplate(
                "Spill kit inspection",
                "Spill kits next to the hazmat cage are inspected every {n} days and the seal number is recorded.",
                "How often are spill kits inspected at {ref}?",
                "How frequently does someone check the leak clean-up equipment at {ref}?",
                "every {n} days",
                {"n": (7, 14, 30)},
            ),
            FactTemplate(
                "Segregation distance",
                "Flammable liquids are stored at least {m} metres away from oxidisers.",
                "How far from oxidisers must flammable liquids be stored at {ref}?",
                "What gap do we keep between fire-risk liquids and oxidizing chemicals at {ref}?",
                "{m} metres",
                {"m": (3, 5, 6)},
            ),
        ],
    },
    "damage-claims": {
        "title": "Carrier Damage Claims",
        "facts": [
            FactTemplate(
                "Filing deadline",
                "Damage claims must be filed with the carrier within {days} days of delivery at {city} ({code}).",
                "Within how many days of delivery must damage claims be filed with the carrier at {ref}?",
                "How long do we have to complain to the transporter about broken goods at {ref}?",
                "{days} days",
                {"days": (3, 5, 7, 9)},
            ),
            FactTemplate(
                "Photo evidence",
                "Each claim needs at least {n} photos showing the outer packaging, the label and the damaged item.",
                "How many photos does a damage claim need at {ref}?",
                "How many pictures should be attached when reporting broken freight at {ref}?",
                "{n} photos",
                {"n": (3, 4, 5, 6)},
            ),
            FactTemplate(
                "Approval threshold",
                "Claims above INR {amt} need approval from the regional finance controller before submission.",
                "Above what amount do damage claims need finance approval at {ref}?",
                "What claim value means the money people must sign off first at {ref}?",
                "INR {amt}",
                {"amt": ("25,000", "50,000", "75,000", "1,00,000")},
            ),
        ],
    },
    "dock-scheduling": {
        "title": "Dock Appointment Scheduling",
        "facts": [
            FactTemplate(
                "Booking lead time",
                "Inbound carriers must book a dock slot at {city} ({code}) at least {hrs} hours in advance.",
                "How many hours in advance must inbound carriers book a dock slot at {ref}?",
                "How early does a trucking company have to reserve an unloading time at {ref}?",
                "{hrs} hours",
                {"hrs": (12, 24, 48)},
            ),
            FactTemplate(
                "Late arrivals",
                "Carriers arriving more than {mins} minutes late are moved to the next free slot.",
                "How late can a carrier arrive before losing its dock slot at {ref}?",
                "What delay makes a truck miss its booked time at {ref}?",
                "{mins} minutes",
                {"mins": (15, 30, 45, 60)},
            ),
            FactTemplate(
                "Detention charges",
                "Detention charges start after {free} free hours at the dock.",
                "After how many free hours do detention charges start at {ref}?",
                "When does the site start paying a waiting fee to the haulier at {ref}?",
                "after {free} hours",
                {"free": (2, 3, 4)},
            ),
        ],
    },
    "cycle-counting": {
        "title": "Cycle Counting",
        "facts": [
            FactTemplate(
                "Count frequency",
                "A-class SKUs at {city} ({code}) are cycle counted every {n} days.",
                "How often are A-class SKUs cycle counted at {ref}?",
                "How regularly do we do a stock check on fast-moving items at {ref}?",
                "every {n} days",
                {"n": (7, 14, 30)},
            ),
            FactTemplate(
                "Variance tolerance",
                "A count variance above {pct} percent of the system quantity triggers a recount.",
                "What count variance triggers a recount at {ref}?",
                "How big a stock mismatch means we have to count again at {ref}?",
                "{pct} percent",
                {"pct": ("0.5", "1", "2")},
            ),
            FactTemplate(
                "Recount owner",
                "Recounts are done by a second counter from the {team} team, never the original counter.",
                "Which team does the recount after a variance at {ref}?",
                "Who checks the stock again when numbers do not match at {ref}?",
                "the {team} team",
                {"team": ("inventory control", "quality", "night shift")},
            ),
        ],
    },
    "cross-dock": {
        "title": "Cross-Dock Operations",
        "facts": [
            FactTemplate(
                "Dwell time",
                "Cross-dock freight at {city} ({code}) must leave within {hrs} hours of arrival.",
                "Within how many hours must cross-dock freight leave at {ref}?",
                "How long can transfer stock sit on the floor before it ships at {ref}?",
                "{hrs} hours",
                {"hrs": (4, 6, 8, 12)},
            ),
            FactTemplate(
                "Heavy pallets",
                "Pallets heavier than {kg} kg are moved only with a counterbalance forklift.",
                "Above what weight must pallets be moved with a counterbalance forklift at {ref}?",
                "From what mass can a load no longer be shifted with a hand jack at {ref}?",
                "{kg} kg",
                {"kg": (800, 1000, 1200)},
            ),
            FactTemplate(
                "Staging lanes",
                "Outbound cross-dock pallets are staged in lanes {lanes}.",
                "Which lanes are outbound cross-dock pallets staged in at {ref}?",
                "Where do we line up transfer stock that is about to ship at {ref}?",
                "lanes {lanes}",
                {"lanes": ("1-4", "5-8", "9-12")},
            ),
        ],
    },
}


def _site_ref(site: Site, rng: random.Random) -> str:
    """How a question refers to a site; users mix codes and city names."""
    return rng.choice((site.code, site.city, f"the {site.city} warehouse", f"site {site.code}"))


def generate(seed: int = 7, sites: tuple[Site, ...] = SITES) -> tuple[list[Document], list[Question]]:
    """Build the corpus and one lexical plus one paraphrased question per fact."""
    if not sites:
        raise ValueError("at least one site is required")
    codes = [s.code for s in sites]
    if len(set(codes)) != len(codes):
        raise ValueError(f"site codes must be unique, got {codes}")

    rng = random.Random(seed)
    docs: list[Document] = []
    questions: list[Question] = []

    for topic_key, topic in TOPICS.items():
        for site in sites:
            doc_id = f"{topic_key}-{site.code.lower()}"
            title = f"{topic['title']} SOP - {site.city} ({site.code})"
            doc = Document(doc_id, title, topic_key, site.code)
            doc.sections.append(
                Section(
                    "Scope",
                    f"This SOP covers {topic['title'].lower()} at the {site.city} warehouse, site code "
                    f"{site.code}. {_FILLER['scope']}",
                )
            )
            for fact in topic["facts"]:
                values = {name: rng.choice(options) for name, options in fact.params.items()}
                fields = {"city": site.city, "code": site.code, **values}
                doc.sections.append(Section(fact.heading, fact.sentence.format(**fields)))
                answer = fact.answer.format(**fields)
                for style, template in (("lexical", fact.lexical_q), ("paraphrase", fact.paraphrase_q)):
                    questions.append(
                        Question(
                            qid=f"q{len(questions) + 1:04d}",
                            question=template.format(ref=_site_ref(site, rng)),
                            gold_doc_id=doc_id,
                            gold_section=fact.heading,
                            answer=answer,
                            topic=topic_key,
                            site_code=site.code,
                            style=style,
                        )
                    )
            doc.sections.append(Section("Records", _FILLER["records"]))
            docs.append(doc)
    return docs, questions

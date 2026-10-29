"""Apply validated CDC events to an SCD Type 2 dimension in DuckDB.

Each lane has one row per version of its rate. A version is open from the
commit time of the change that created it (inclusive) to the commit time of the
next change to that lane (exclusive). The open version of a live lane has
``valid_to = 9999-12-31`` and ``is_current = true``; a deleted lane has no
current row.

The merge is set-based SQL so that a batch with several changes to the same
lane produces the right chain of versions in one pass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import duckdb

from .events import ChangeEvent
from .validation import event_record, validate_batch

OPEN_END = datetime(9999, 12, 31)

SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_rate_card (
    lane_id      VARCHAR   NOT NULL,
    carrier      VARCHAR   NOT NULL,
    origin       VARCHAR   NOT NULL,
    destination  VARCHAR   NOT NULL,
    rate_per_kg  DOUBLE    NOT NULL,
    min_charge   DOUBLE    NOT NULL,
    currency     VARCHAR   NOT NULL,
    row_hash     VARCHAR   NOT NULL,
    valid_from   TIMESTAMP NOT NULL,
    valid_to     TIMESTAMP NOT NULL,
    is_current   BOOLEAN   NOT NULL,
    source_lsn   BIGINT    NOT NULL
);
CREATE TABLE IF NOT EXISTS load_state (
    id             INTEGER PRIMARY KEY,
    watermark_lsn  BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS applied_events (
    lsn           BIGINT PRIMARY KEY,
    payload_hash  VARCHAR NOT NULL,
    batch_id      VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS load_log (
    batch_id                 VARCHAR,
    received                 INTEGER,
    rejected                 INTEGER,
    duplicates               INTEGER,
    below_watermark          INTEGER,
    no_op                    INTEGER,
    versions_opened          INTEGER,
    versions_closed          INTEGER,
    watermark_after          BIGINT,
    loaded_at                TIMESTAMP  -- UTC
);
CREATE TABLE IF NOT EXISTS quarantine (
    batch_id  VARCHAR,
    lsn       BIGINT,
    reason    VARCHAR,
    record    VARCHAR
);
"""

STAGE = """
CREATE OR REPLACE TEMP TABLE stg_events (
    lsn BIGINT, op VARCHAR, commit_ts TIMESTAMP, lane_id VARCHAR,
    carrier VARCHAR, origin VARCHAR, destination VARCHAR,
    rate_per_kg DOUBLE, min_charge DOUBLE, currency VARCHAR
)
"""

# Keep only the events that change the state of their lane: a create/update
# whose values differ from the lane's previous state, or a delete of a live lane.
# The previous state of the first event in the batch is the lane's open version.
EFFECTIVE = """
CREATE OR REPLACE TEMP TABLE stg_effective AS
WITH hashed AS (
    SELECT s.*,
           CASE WHEN op = 'd' THEN NULL
                ELSE md5(concat_ws('|', carrier, origin, destination,
                                   rate_per_kg::VARCHAR, min_charge::VARCHAR, currency))
           END AS row_hash
    FROM stg_events s
),
with_prev AS (
    SELECT h.*,
           lag(h.row_hash, 1, d.row_hash) OVER (PARTITION BY h.lane_id ORDER BY h.lsn) AS prev_hash,
           row_number() OVER (PARTITION BY h.lane_id ORDER BY h.lsn) AS seq
    FROM hashed h
    LEFT JOIN dim_rate_card d ON d.lane_id = h.lane_id AND d.is_current
)
SELECT * FROM with_prev
WHERE (op = 'd' AND prev_hash IS NOT NULL)
   OR (op <> 'd' AND row_hash IS DISTINCT FROM prev_hash)
"""

# A change committed at the same instant as the lane's open version replaces it:
# no point in time ever saw the earlier values, so keeping them would leave a
# zero-length version in the history.
REPLACE_SAME_INSTANT = """
DELETE FROM dim_rate_card AS d
USING (SELECT lane_id, min(commit_ts) AS first_ts FROM stg_effective GROUP BY lane_id) f
WHERE d.lane_id = f.lane_id AND d.is_current AND d.valid_from = f.first_ts
"""

CLOSE_CURRENT = """
UPDATE dim_rate_card AS d
SET valid_to = f.first_ts, is_current = false
FROM (SELECT lane_id, min(commit_ts) AS first_ts FROM stg_effective GROUP BY lane_id) f
WHERE d.lane_id = f.lane_id AND d.is_current
"""

INSERT_VERSIONS = """
INSERT INTO dim_rate_card
SELECT lane_id, carrier, origin, destination, rate_per_kg, min_charge, currency, row_hash,
       commit_ts AS valid_from,
       coalesce(next_ts, ?) AS valid_to,
       next_ts IS NULL AS is_current,
       lsn AS source_lsn
FROM (
    SELECT *, lead(commit_ts) OVER (PARTITION BY lane_id ORDER BY lsn) AS next_ts
    FROM stg_effective
)
WHERE op <> 'd'
"""


@dataclass(frozen=True)
class BatchReport:
    batch_id: str
    received: int
    rejected: int
    duplicates: int
    below_watermark: int
    no_op: int
    versions_opened: int
    versions_closed: int
    watermark_after: int
    versions_replaced: int = 0


class RateCardLoader:
    """Loads CDC batches into ``dim_rate_card`` on a DuckDB connection."""

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self.con.execute(SCHEMA)
        self.con.execute(
            "INSERT INTO load_state SELECT 1, 0 WHERE NOT EXISTS (SELECT 1 FROM load_state)"
        )

    @property
    def watermark(self) -> int:
        return self.con.execute("SELECT watermark_lsn FROM load_state WHERE id = 1").fetchone()[0]

    def load_batch(self, records: list[dict], batch_id: str) -> BatchReport:
        """Validate and apply one delivery batch atomically.

        Every event that passes validation is recorded in ``applied_events``
        with a hash of its payload. An event at or below the watermark is
        skipped when it matches an applied event exactly (a redelivery or a
        replay), which makes loading the same batch twice a no-op. If its LSN
        was never applied (a late event) or was applied with a different
        payload, it is quarantined so the gap is visible instead of silent.
        """
        result = validate_batch(records)
        watermark = self.watermark
        fresh = [e for e in result.valid if e.lsn > watermark]
        stale = [e for e in result.valid if e.lsn <= watermark]
        below_watermark, stale_rejects = self._check_below_watermark(stale, watermark)
        fresh, out_of_order = self._check_commit_order(fresh)
        ledger = list(fresh)
        fresh, superseded = _keep_last_per_instant(fresh)
        rejected = result.rejected + stale_rejects + out_of_order

        self.con.execute("BEGIN TRANSACTION")
        try:
            self._quarantine(batch_id, rejected)
            if ledger:
                self.con.executemany(
                    "INSERT INTO applied_events VALUES (?, ?, ?)",
                    [[e.lsn, payload_hash(e), batch_id] for e in ledger],
                )
            self.con.execute(STAGE)
            if fresh:
                self.con.executemany(
                    "INSERT INTO stg_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [_row(e) for e in fresh],
                )
            self.con.execute(EFFECTIVE)
            effective = self.con.execute("SELECT count(*) FROM stg_effective").fetchone()[0]
            replaced = _rowcount(self.con.execute(REPLACE_SAME_INSTANT))
            closed = _rowcount(self.con.execute(CLOSE_CURRENT))
            opened = _rowcount(self.con.execute(INSERT_VERSIONS, [OPEN_END]))
            # Versions opened and superseded inside this batch were also closed.
            closed += opened - self._lanes_current_from_batch()
            new_watermark = max([watermark] + [e.lsn for e in result.valid])
            self.con.execute("UPDATE load_state SET watermark_lsn = ? WHERE id = 1", [new_watermark])
            report = BatchReport(
                batch_id=batch_id,
                received=len(records),
                rejected=len(rejected),
                duplicates=result.duplicates,
                below_watermark=below_watermark,
                no_op=len(fresh) - effective + superseded,
                versions_replaced=replaced,
                versions_opened=opened,
                versions_closed=closed,
                watermark_after=new_watermark,
            )
            self.con.execute(
                "INSERT INTO load_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    report.batch_id,
                    report.received,
                    report.rejected,
                    report.duplicates,
                    report.below_watermark,
                    report.no_op,
                    report.versions_opened,
                    report.versions_closed,
                    report.watermark_after,
                    datetime.now(timezone.utc).replace(tzinfo=None),
                ],
            )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return report

    def _check_below_watermark(
        self, events: list[ChangeEvent], watermark: int
    ) -> tuple[int, list[tuple[dict, str]]]:
        """Split events at or below the watermark into redeliveries and rejects.

        Returns the number of exact redeliveries and the rejects with reasons.
        """
        if not events:
            return 0, []
        lsns = [e.lsn for e in events]
        applied = dict(
            self.con.execute(
                "SELECT lsn, payload_hash FROM applied_events WHERE list_contains(?, lsn)", [lsns]
            ).fetchall()
        )
        quarantined = {
            lsn
            for (lsn,) in self.con.execute(
                "SELECT DISTINCT lsn FROM quarantine WHERE list_contains(?, lsn)", [lsns]
            ).fetchall()
        }
        redelivered, rejected = 0, []
        for event in events:
            known = applied.get(event.lsn)
            if known == payload_hash(event):
                redelivered += 1
            elif known is not None:
                rejected.append(
                    (event_record(event), f"lsn {event.lsn} was already applied with a different payload")
                )
            elif event.lsn in quarantined:
                rejected.append(
                    (event_record(event), f"lsn {event.lsn} was quarantined in an earlier batch")
                )
            else:
                rejected.append(
                    (
                        event_record(event),
                        f"late event: lsn {event.lsn} is below the watermark {watermark} "
                        "and was never applied",
                    )
                )
        return redelivered, rejected

    def _check_commit_order(
        self, events: list[ChangeEvent]
    ) -> tuple[list[ChangeEvent], list[tuple[dict, str]]]:
        """Reject events committed before the previous change to their lane.

        LSN order is the source's order of changes, so a later LSN with an
        earlier commit time means the clock or the feed is wrong. Applying it
        would close the open version before it started. The previous change is
        the latest version boundary in the dimension or, inside this batch, the
        last accepted event for the lane.
        """
        lanes = sorted({e.lane_id for e in events})
        last_ts: dict[str, datetime] = {}
        if lanes:
            last_ts = dict(
                self.con.execute(
                    "SELECT lane_id, max(CASE WHEN is_current THEN valid_from ELSE valid_to END) "
                    "FROM dim_rate_card WHERE list_contains(?, lane_id) GROUP BY lane_id",
                    [lanes],
                ).fetchall()
            )
        accepted, rejected = [], []
        for event in events:  # already sorted by LSN
            previous = last_ts.get(event.lane_id)
            if previous is not None and event.commit_ts < previous:
                rejected.append(
                    (
                        event_record(event),
                        f"commit_ts {event.commit_ts.isoformat()} is earlier than the lane's "
                        f"previous change at {previous.isoformat()}",
                    )
                )
                continue
            accepted.append(event)
            last_ts[event.lane_id] = event.commit_ts
        return accepted, rejected

    def _lanes_current_from_batch(self) -> int:
        return self.con.execute(
            "SELECT count(*) FROM dim_rate_card d JOIN (SELECT DISTINCT lane_id FROM stg_effective) f "
            "USING (lane_id) WHERE d.is_current"
        ).fetchone()[0]

    def _quarantine(self, batch_id: str, rejected: list[tuple[dict, str]]) -> None:
        if not rejected:
            return
        rows = []
        for record, reason in rejected:
            lsn = record.get("lsn")
            lsn = lsn if isinstance(lsn, int) and not isinstance(lsn, bool) else None
            rows.append([batch_id, lsn, reason, json.dumps(record, sort_keys=True, default=str)])
        self.con.executemany("INSERT INTO quarantine VALUES (?, ?, ?, ?)", rows)


def payload_hash(event: ChangeEvent) -> str:
    """Stable hash of an event's full content, used to recognise redeliveries."""
    return hashlib.md5(event.to_json().encode("utf-8")).hexdigest()


def _keep_last_per_instant(events: list[ChangeEvent]) -> tuple[list[ChangeEvent], int]:
    """Keep only the highest LSN among a lane's changes committed at the same instant.

    The source can change a row twice in one transaction; both changes share a
    commit time and only the final state was ever visible.
    """
    last: dict[tuple[str, datetime], ChangeEvent] = {}
    for event in events:
        last[(event.lane_id, event.commit_ts)] = event  # events are sorted by LSN
    kept = sorted(last.values(), key=lambda e: e.lsn)
    return kept, len(events) - len(kept)


def _rowcount(cursor: duckdb.DuckDBPyConnection) -> int:
    row = cursor.fetchone()
    return row[0] if row else 0


def _row(event: ChangeEvent) -> list:
    return [
        event.lsn,
        event.op,
        event.commit_ts,
        event.lane_id,
        event.carrier,
        event.origin,
        event.destination,
        event.rate_per_kg,
        event.min_charge,
        event.currency,
    ]

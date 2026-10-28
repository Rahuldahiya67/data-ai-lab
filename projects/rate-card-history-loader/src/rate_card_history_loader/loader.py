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

import json
from dataclasses import dataclass
from datetime import datetime

import duckdb

from .events import ChangeEvent
from .validation import validate_batch

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
    loaded_at                TIMESTAMP
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

        Events at or below the watermark were applied by an earlier batch (a
        redelivery or a replay) and are skipped, which makes loading the same
        batch twice a no-op.
        """
        result = validate_batch(records)
        watermark = self.watermark
        fresh = [e for e in result.valid if e.lsn > watermark]
        conflicts = sum(1 for _, reason in result.rejected if "conflicting payloads" in reason)
        parsed = len(records) - (len(result.rejected) - conflicts)
        duplicates = parsed - conflicts - len(result.valid)

        self.con.execute("BEGIN TRANSACTION")
        try:
            self._quarantine(batch_id, result.rejected)
            self.con.execute(STAGE)
            if fresh:
                self.con.executemany(
                    "INSERT INTO stg_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [_row(e) for e in fresh],
                )
            self.con.execute(EFFECTIVE)
            effective = self.con.execute("SELECT count(*) FROM stg_effective").fetchone()[0]
            closed = self.con.execute(
                "SELECT count(*) FROM dim_rate_card d JOIN (SELECT DISTINCT lane_id FROM stg_effective) f "
                "USING (lane_id) WHERE d.is_current"
            ).fetchone()[0]
            self.con.execute(CLOSE_CURRENT)
            before = self._version_count()
            self.con.execute(INSERT_VERSIONS, [OPEN_END])
            opened = self._version_count() - before
            # Versions opened and superseded inside this batch were also closed.
            closed += opened - self._lanes_current_from_batch()
            new_watermark = max([watermark] + [e.lsn for e in fresh])
            self.con.execute("UPDATE load_state SET watermark_lsn = ? WHERE id = 1", [new_watermark])
            report = BatchReport(
                batch_id=batch_id,
                received=len(records),
                rejected=len(result.rejected),
                duplicates=duplicates,
                below_watermark=len(result.valid) - len(fresh),
                no_op=len(fresh) - effective,
                versions_opened=opened,
                versions_closed=closed,
                watermark_after=new_watermark,
            )
            self.con.execute(
                "INSERT INTO load_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, now()::TIMESTAMP)",
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
                ],
            )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return report

    def _version_count(self) -> int:
        return self.con.execute("SELECT count(*) FROM dim_rate_card").fetchone()[0]

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

"""Inventory of the WRDS tables this project could use.

Only metadata is collected: table names, column names and types, date ranges and
the codes of a few categorical columns. No security-level row is read or stored.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

Query = Callable[[str], pd.DataFrame]

LIBRARY_GROUPS = {
    "crsp": r"^crsp",
    "ccm": r"ccm",
    "compustat": r"^comp",
    "ibes": r"ibes",
    "factors": r"^ff",
    "rates": r"^frb",
    "cboe": r"^cboe",
    "links": r"^wrdsapps_link",
}

PRIORITY_TABLES = {
    "crsp": [
        "dsf_v2", "msf_v2", "stkdlysecuritydata", "stksecurityinfohist", "stkissuerinfohist",
        "stkdelists", "stkdistributions", "stkshares", "stocknames_v2", "dsp500list_v2",
        "inddlyseriesdata", "dsf", "dsi", "dsenames", "dsedelist", "dsedist",
        "ccmxpf_lnkhist", "ccmxpf_linktable",
    ],
    "ccm": ["ccmxpf_lnkhist", "ccmxpf_linktable"],
    "compustat": ["funda", "fundq", "company", "names", "security"],
    "ibes": ["statsum_epsus", "statsumu_epsus", "det_epsus", "actu_epsus", "surpsum", "idsum"],
    "factors": ["factors_daily", "fivefactors_daily"],
    "rates": ["rates_daily"],
    "cboe": ["cboe"],
    "links": ["ibcrsphist"],
}

# Most informative first: the first one found in a table gives its date range.
DATE_COLUMNS = (
    "dlycaldt", "caldt", "date", "datadate", "statpers", "anndats", "actdats",
    "secinfoenddt", "delistingdate", "disexdt", "linkdt", "sdate",
)

# Columns the daily panel needs, used to flag gaps early.
CRITICAL_COLUMNS = {
    "dsf_v2": [
        "permno", "permco", "dlycaldt", "dlyret", "dlyretx", "dlyprc", "dlyvol", "dlybid", "dlyask",
        "dlycap", "shrout", "siccd", "ticker", "primaryexch", "sharetype", "securitytype",
        "securitysubtype", "usincflg", "issuertype", "conditionaltype", "tradingstatusflg",
    ],
    "dsf": ["permno", "permco", "date", "ret", "retx", "prc", "vol", "bid", "ask", "shrout", "cfacpr"],
}

# Categorical codes needed to write the common-stock filter.
VALUE_PROBES = {
    "stksecurityinfohist": [
        "sharetype", "securitytype", "securitysubtype", "usincflg",
        "issuertype", "primaryexch", "conditionaltype", "tradingstatusflg",
    ],
    "stkdelists": ["delreasontype", "delstatustype", "delpaymenttype"],
}

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def _quote_list(names: list[str]) -> str:
    unsafe = [name for name in names if not _SAFE_NAME.match(name)]
    if unsafe:
        raise ValueError(f"Unexpected identifiers: {unsafe}")
    return ", ".join(f"'{name}'" for name in names)


def group_libraries(libraries: set[str]) -> dict[str, list[str]]:
    """Group visible libraries, shortest (usually canonical) name first."""
    return {
        group: sorted((lib for lib in libraries if re.search(pattern, lib)), key=lambda x: (len(x), x))
        for group, pattern in LIBRARY_GROUPS.items()
    }


def fetch_columns(query: Query, libraries: list[str]) -> dict[str, dict[str, list[dict]]]:
    """Return {library: {table: [{name, type}, ...]}} for tables visible to this account."""
    if not libraries:
        return {}
    rows = query(
        "SELECT table_schema, table_name, column_name, data_type "
        "FROM information_schema.columns "
        f"WHERE table_schema IN ({_quote_list(libraries)}) "
        "ORDER BY table_schema, table_name, ordinal_position"
    )
    columns: dict[str, dict[str, list[dict]]] = {lib: {} for lib in libraries}
    for row in rows.itertuples(index=False):
        columns[row.table_schema].setdefault(row.table_name, []).append(
            {"name": row.column_name, "type": row.data_type}
        )
    return columns


def _attempt(query: Query, sql: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return query(sql), None
    except Exception as exc:  # A denied or slow table must not stop the inventory.
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"


def probe_table(query: Query, library: str, table: str, table_columns: list[dict]) -> dict:
    """Test read access to a table and measure its date range."""
    names = [column["name"] for column in table_columns]
    probe = {
        "library": library,
        "n_columns": len(names),
        "columns": table_columns,
        "missing_critical": [c for c in CRITICAL_COLUMNS.get(table, []) if c not in names],
    }

    _, error = _attempt(query, f"SELECT 1 AS ok FROM {library}.{table} LIMIT 1")
    probe["readable"] = error is None
    if error:
        probe["error"] = error
        return probe

    date_column = next((c for c in DATE_COLUMNS if c in names), None)
    probe["date_column"] = date_column
    if date_column:
        dates, error = _attempt(
            query, f"SELECT min({date_column}) AS min_date, max({date_column}) AS max_date FROM {library}.{table}"
        )
        if error:
            probe["date_error"] = error
        else:
            probe["min_date"] = str(dates["min_date"].iloc[0])
            probe["max_date"] = str(dates["max_date"].iloc[0])

    probes = {}
    for column in VALUE_PROBES.get(table, []):
        if column not in names:
            continue
        counts, error = _attempt(
            query,
            f"SELECT {column} AS value, count(*) AS n FROM {library}.{table} "
            f"GROUP BY {column} ORDER BY n DESC LIMIT 40",
        )
        probes[column] = {"error": error} if error else {
            str(row.value): int(row.n) for row in counts.itertuples(index=False)
        }
    if probes:
        probe["value_counts"] = probes
    return probe


def collect_inventory(query: Query, libraries: set[str]) -> dict:
    """Build the full inventory from the libraries visible to this account."""
    groups = group_libraries(libraries)
    watched = sorted({lib for group in groups.values() for lib in group})
    columns = fetch_columns(query, watched)

    priority: dict[str, dict] = {}
    for group, tables in PRIORITY_TABLES.items():
        priority[group] = {}
        for table in tables:
            holders = [lib for lib in groups[group] if table in columns.get(lib, {})]
            if not holders:
                priority[group][table] = {"found": False}
                continue
            probe = probe_table(query, holders[0], table, columns[holders[0]][table])
            priority[group][table] = {"found": True, "also_in": holders[1:], **probe}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "all_libraries": sorted(libraries),
        "groups": groups,
        "tables": {lib: sorted(tables) for lib, tables in columns.items()},
        "priority": priority,
    }


def render_markdown(report: dict) -> str:
    """Summarize an inventory for a human reader."""
    lines = [f"# WRDS inventory ({report['generated_at']})", ""]
    lines += [f"{len(report['all_libraries'])} visible libraries.", "", "## Libraries by group", ""]
    for group, libraries in report["groups"].items():
        lines.append(f"- **{group}**: {', '.join(libraries) if libraries else 'none'}")

    for group, tables in report["priority"].items():
        lines += ["", f"## {group}", "", "| Table | Library | Readable | Columns | Start | End | Missing columns |"]
        lines.append("|---|---|---|---:|---|---|---|")
        for table, info in tables.items():
            if not info["found"]:
                lines.append(f"| `{table}` | missing | | | | | |")
                continue
            missing = ", ".join(info.get("missing_critical", []))
            lines.append(
                f"| `{table}` | {info['library']} | {'yes' if info['readable'] else 'NO'} "
                f"| {info['n_columns']} | {info.get('min_date', '')} | {info.get('max_date', '')} | {missing} |"
            )
    return "\n".join(lines) + "\n"

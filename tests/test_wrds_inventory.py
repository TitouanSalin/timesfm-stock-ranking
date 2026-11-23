"""Run the inventory logic against a small DuckDB catalog shaped like WRDS."""

import duckdb
import pytest

from qef.wrds_inventory import collect_inventory, render_markdown


@pytest.fixture()
def report():
    con = duckdb.connect()
    con.execute("CREATE SCHEMA crsp")
    con.execute("CREATE SCHEMA crsp_a_stock")
    con.execute("CREATE SCHEMA ff")
    con.execute("CREATE SCHEMA taq_2014")
    con.execute(
        "CREATE TABLE crsp.dsf_v2 AS "
        "SELECT 10001 AS permno, DATE '2000-01-03' AS dlycaldt, 0.01 AS dlyret "
        "UNION ALL SELECT 10001, DATE '2025-12-31', 0.02"
    )
    con.execute("CREATE TABLE crsp_a_stock.dsf_v2 AS SELECT * FROM crsp.dsf_v2")
    con.execute(
        "CREATE TABLE crsp.stksecurityinfohist AS "
        "SELECT 10001 AS permno, 'NS' AS sharetype, DATE '2030-01-01' AS secinfoenddt "
        "UNION ALL SELECT 10002, 'NS', DATE '2030-01-01' UNION ALL SELECT 10003, 'AD', DATE '2030-01-01'"
    )
    con.execute("CREATE TABLE ff.factors_daily AS SELECT DATE '1990-01-02' AS date, 0.01 AS mktrf")

    def query(sql: str):
        return con.execute(sql).fetchdf()

    return collect_inventory(query, {"crsp", "crsp_a_stock", "ff", "taq_2014"})


def test_groups_exclude_unrelated_libraries(report):
    assert report["groups"]["crsp"] == ["crsp", "crsp_a_stock"]
    watched = {lib for libs in report["groups"].values() for lib in libs}
    assert "taq_2014" not in watched


def test_priority_table_is_probed_on_shortest_library(report):
    info = report["priority"]["crsp"]["dsf_v2"]
    assert info["found"] and info["readable"]
    assert info["library"] == "crsp"
    assert info["also_in"] == ["crsp_a_stock"]
    assert info["date_column"] == "dlycaldt"
    assert info["min_date"].startswith("2000-01-03")
    assert info["max_date"].startswith("2025-12-31")


def test_missing_columns_and_tables_are_flagged(report):
    assert "dlybid" in report["priority"]["crsp"]["dsf_v2"]["missing_critical"]
    assert "permno" not in report["priority"]["crsp"]["dsf_v2"]["missing_critical"]
    assert report["priority"]["crsp"]["dsf"] == {"found": False}


def test_categorical_codes_are_counted(report):
    counts = report["priority"]["crsp"]["stksecurityinfohist"]["value_counts"]["sharetype"]
    assert counts == {"NS": 2, "AD": 1}


def test_a_failing_table_does_not_stop_the_inventory():
    con = duckdb.connect()
    con.execute("CREATE SCHEMA ff")
    con.execute("CREATE TABLE ff.factors_daily AS SELECT DATE '1990-01-02' AS date")

    def query(sql: str):
        if "factors_daily" in sql and "information_schema" not in sql:
            raise PermissionError("permission denied")
        return con.execute(sql).fetchdf()

    report = collect_inventory(query, {"ff"})
    info = report["priority"]["factors"]["factors_daily"]
    assert info["found"] and not info["readable"]
    assert "permission denied" in info["error"]


def test_markdown_lists_every_group(report):
    text = render_markdown(report)
    assert "## crsp" in text and "`dsf_v2`" in text and "missing" in text

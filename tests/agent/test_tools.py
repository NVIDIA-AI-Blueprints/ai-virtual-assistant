# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the database tools in src/agent/tools.py.

psycopg2.connect is replaced by a fake connection whose cursor records every
execute(sql, params) call and runs the statement on an in-memory SQLite copy of
the customer_data table, so no Postgres server, config file or NIM is needed.

Run from the repository root:
    pip install langchain-core psycopg2-binary requests pytest
    python -m pytest tests/agent
"""

import importlib
import sqlite3
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# (customer_id, order_id, product_name, order_date, order_status, return_status)
ORDERS = [
    (1, 10, "RTX 4090", "2024-10-20", "Delivered", None),
    (1, 11, "DGX Spark", "2024-10-25", "Delivered", None),
    (2, 20, "Jetson Orin Nano", "2024-10-22", "Delivered", None),
    (3, 30, "RTX 5080", "2024-10-28", "Shipped", None),
]


class FakeCursor:
    def __init__(self, db, calls):
        self._cur = db.cursor()
        self._calls = calls

    def execute(self, sql, params=None):
        self._calls.append((sql, params))
        # psycopg2 uses %s placeholders, sqlite uses ?
        self._cur.execute(sql.replace("%s", "?"), params or ())

    def _row(self, values):
        return dict(zip([d[0] for d in self._cur.description], values))

    def fetchone(self):
        values = self._cur.fetchone()
        return self._row(values) if values else None

    def fetchall(self):
        return [self._row(values) for values in self._cur.fetchall()]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, db, calls):
        self._db = db
        self._calls = calls

    def cursor(self, cursor_factory=None):
        return FakeCursor(self._db, self._calls)

    def commit(self):
        self._db.commit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.execute("ATTACH DATABASE ':memory:' AS public")
    conn.execute(
        """CREATE TABLE public.customer_data (
            customer_id INTEGER NOT NULL, order_id INTEGER NOT NULL, product_name TEXT NOT NULL,
            order_date TEXT NOT NULL, order_status TEXT, quantity INTEGER, order_amount REAL,
            return_status TEXT, return_start_date TEXT, return_received_date TEXT,
            return_completed_date TEXT, return_reason TEXT, notes TEXT)"""
    )
    conn.executemany(
        "INSERT INTO public.customer_data (customer_id, order_id, product_name, order_date, "
        "order_status, return_status, quantity, order_amount) VALUES (?, ?, ?, ?, ?, ?, 1, 1.0)",
        ORDERS,
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def calls():
    return []


@pytest.fixture
def tools(monkeypatch, db, calls):
    """Import a fresh copy of src.agent.tools wired to the fake database."""
    utils_stub = types.ModuleType("src.common.utils")
    utils_stub.get_prompts = dict
    utils_stub.get_config = lambda: types.SimpleNamespace(
        database=types.SimpleNamespace(url="postgres:5432")
    )
    monkeypatch.setitem(sys.modules, "src.common.utils", utils_stub)
    monkeypatch.delitem(sys.modules, "src.agent.tools", raising=False)
    module = importlib.import_module("src.agent.tools")
    monkeypatch.setattr(
        module.psycopg2, "connect", lambda **kwargs: FakeConnection(db, calls)
    )
    yield module
    sys.modules.pop("src.agent.tools", None)


def return_statuses(db):
    return db.execute(
        "SELECT order_id, return_status FROM public.customer_data ORDER BY order_id"
    ).fetchall()


UNCHANGED = [(10, None), (11, None), (20, None), (30, None)]


@pytest.mark.parametrize(
    "args",
    [
        {"user_id": "1", "current_product": "RTX 4090", "order_id": "10' OR '1'='1"},
        {"user_id": "1' OR '1'='1", "current_product": "RTX 4090", "order_id": "10"},
        {"user_id": "1", "current_product": "x' OR 'a'='a", "order_id": "11"},
    ],
)
def test_update_return_binds_arguments_as_parameters(tools, db, calls, args):
    result = tools.update_return.invoke(args)

    assert result.startswith("No matching order found")
    assert return_statuses(db) == UNCHANGED
    assert calls, "update_return did not query the database"
    expected = (args["user_id"], args["current_product"], args["order_id"])
    payload = next(value for value in expected if "'" in value)
    for sql, params in calls:
        assert params == expected
        assert payload not in sql


def test_get_purchase_history_binds_user_id(tools, calls):
    payload = "1 OR 1=1"

    assert tools.get_purchase_history.invoke({"user_id": payload}) == []

    ((sql, params),) = calls
    assert params == (payload,)
    assert payload not in sql


def test_update_return_updates_only_the_named_order(tools, db):
    result = tools.update_return.invoke(
        {"user_id": "1", "current_product": "RTX 4090", "order_id": "10"}
    )

    assert result == "Return status for order_id 10 has been updated to 'Requested'."
    assert return_statuses(db) == [
        (10, "Requested"),
        (11, None),
        (20, None),
        (30, None),
    ]


@pytest.mark.parametrize(
    "read_tool", ["get_purchase_history", "get_recent_return_details"]
)
def test_read_after_update_is_not_stale(tools, read_tool):
    read = getattr(tools, read_tool)

    def status_of_order_10():
        orders = read.invoke({"user_id": "1"})
        return next(o["return_status"] for o in orders if o["order_id"] == 10)

    assert status_of_order_10() is None
    tools.update_return.invoke(
        {"user_id": "1", "current_product": "RTX 4090", "order_id": "10"}
    )
    assert status_of_order_10() == "Requested"


def test_update_return_checks_the_database_on_every_call(tools, db):
    args = {"user_id": "1", "current_product": "RTX 4090", "order_id": "10"}
    assert tools.update_return.invoke(args).startswith("Return status for order_id 10")

    db.execute("DELETE FROM public.customer_data WHERE order_id = 10")
    db.commit()

    assert tools.update_return.invoke(args).startswith("No matching order found")


def test_structured_rag_queries_the_retriever_on_every_call(tools, monkeypatch):
    answers = iter(["return_status: None", "return_status: Requested"])

    def fake_post(url, json):
        body = {"chunks": [{"content": next(answers)}]}
        return types.SimpleNamespace(status_code=200, json=lambda: body)

    monkeypatch.setattr(tools.requests, "post", fake_post)
    args = {"query": "status of my RTX 4090 return", "user_id": "1"}

    assert tools.structured_rag.invoke(args) == "return_status: None"
    assert tools.structured_rag.invoke(args) == "return_status: Requested"

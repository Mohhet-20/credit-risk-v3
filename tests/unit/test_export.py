"""Unit tests for `reporting/export.py`'s CSV/Excel helpers."""

from __future__ import annotations

import csv
import io

import openpyxl

from reporting.export import rows_to_csv_bytes, rows_to_excel_bytes


def test_rows_to_csv_bytes_round_trips():
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    data = rows_to_csv_bytes(rows)
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    parsed = list(reader)
    assert parsed == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def test_rows_to_csv_bytes_empty_input_returns_empty_bytes():
    assert rows_to_csv_bytes([]) == b""


def test_rows_to_csv_bytes_missing_key_fills_blank():
    rows = [{"a": 1, "b": "x"}, {"a": 2}]
    data = rows_to_csv_bytes(rows)
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    parsed = list(reader)
    assert parsed[1]["b"] == ""


def test_rows_to_excel_bytes_round_trips():
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    data = rows_to_excel_bytes(rows, sheet_name="Test")
    wb = openpyxl.load_workbook(io.BytesIO(data))
    ws = wb["Test"]
    assert [c.value for c in ws[1]] == ["a", "b"]
    assert [c.value for c in ws[2]] == [1, "x"]
    assert [c.value for c in ws[3]] == [2, "y"]


def test_rows_to_excel_bytes_header_is_bold():
    data = rows_to_excel_bytes([{"a": 1}])
    wb = openpyxl.load_workbook(io.BytesIO(data))
    ws = wb.active
    assert ws["A1"].font.bold is True


def test_rows_to_excel_bytes_empty_input_produces_valid_workbook():
    data = rows_to_excel_bytes([])
    wb = openpyxl.load_workbook(io.BytesIO(data))
    assert wb.active is not None

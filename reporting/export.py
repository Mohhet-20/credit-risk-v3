"""
Tabular export helpers: CSV and Excel (.xlsx), shared by every "download
this table" endpoint across the API (prediction history, audit logs, model
comparison, data-quality history) rather than each router hand-rolling its
own serialization.

Design decision: both helpers take the exact same input shape --
`list[dict]` with a fixed column order determined by the FIRST row's keys
(every row is expected to share the same keys; a row missing a key gets a
blank cell rather than raising, since a partially-populated export is more
useful than a 500). This keeps the helpers trivially reusable across very
different domain objects (predictions, audit logs, model versions) without
a shared base schema -- callers just build a list of flat dicts however
makes sense for their domain and hand it to `rows_to_csv_bytes()` /
`rows_to_excel_bytes()`.

Design decision: Excel export uses `openpyxl` directly (already a
`pandas` dependency, but used directly here rather than via
`DataFrame.to_excel()`) so a header row can get bold styling and
auto-sized columns without pulling `pandas` into what is otherwise a
plain list-of-dicts -> bytes conversion -- keeps this module's only
dependency the one actually needed for the richer output.
"""

from __future__ import annotations

import csv
import io


def rows_to_csv_bytes(rows: list[dict]) -> bytes:
    """Returns UTF-8 CSV bytes. An empty `rows` list produces an empty
    (zero-byte) file rather than raising -- an admin exporting an empty
    audit log should get a valid, openable empty CSV, not an error."""
    if not rows:
        return b""
    buf = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return buf.getvalue().encode("utf-8")


def rows_to_excel_bytes(rows: list[dict], sheet_name: str = "Report") -> bytes:
    """Returns .xlsx bytes with a bold header row and auto-sized columns.
    Same empty-input behavior as `rows_to_csv_bytes()`: an empty `rows`
    produces a valid (header-only) workbook, not an error."""
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name[:31]  # Excel sheet-name length limit

    if rows:
        fieldnames = list(rows[0].keys())
        ws.append(fieldnames)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for row in rows:
            ws.append([row.get(k, "") for k in fieldnames])

        for i, name in enumerate(fieldnames, start=1):
            max_len = max([len(str(name))] + [len(str(row.get(name, ""))) for row in rows])
            ws.column_dimensions[get_column_letter(i)].width = min(max_len + 2, 60)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

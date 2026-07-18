"""
report_export
=============
Builds a professional Excel risk report from a valued book, in memory, for the
Streamlit download button. Kept in the app layer (not the engine): it consumes
already-computed risk numbers and lays them out for a treasury desk.

Style follows desk conventions: a consistent font, thousands separators,
parentheses for negatives, units in headers, a colour-coded limits block, and
separate sheets for summary, positions and risk detail.
"""
from __future__ import annotations

import io
from datetime import datetime


# Desk palette
_INK = "#1A1A2E"
_HEADER_BG = "#1F3A5F"
_ACCENT_BG = "#2E5A88"
_OK_BG = "#1B5E20"
_BREACH_BG = "#8B1A1A"
_LIGHT = "#F2F4F7"


def build_excel_report(*, book_rows: list[dict], summary: dict,
                       risk_detail: dict) -> bytes:
    """
    Assemble the workbook and return the raw bytes.

    book_rows: list of position dicts (pair, side, notional, rate, tenor, mtm).
    summary: dict with book_value, notional, the 5 VaR figures, ES, limits.
    risk_detail: dict with dv01_by_ccy, dv01_by_tenor, liquidity, stress rows.
    """
    import xlsxwriter

    buffer = io.BytesIO()
    wb = xlsxwriter.Workbook(buffer, {"in_memory": True})

    # ---- formats ----
    f_title = wb.add_format({"bold": True, "font_size": 16, "font_name": "Arial",
                             "font_color": _INK})
    f_sub = wb.add_format({"font_size": 9, "italic": True, "font_name": "Arial",
                           "font_color": "#667085"})
    f_hdr = wb.add_format({"bold": True, "font_name": "Arial", "font_color": "white",
                           "bg_color": _HEADER_BG, "border": 1, "align": "center",
                           "valign": "vcenter"})
    f_label = wb.add_format({"bold": True, "font_name": "Arial", "bg_color": _LIGHT,
                             "border": 1})
    f_num = wb.add_format({"font_name": "Arial", "num_format": "#,##0;(#,##0)",
                           "border": 1})
    f_num2 = wb.add_format({"font_name": "Arial", "num_format": "#,##0.00;(#,##0.00)",
                            "border": 1})
    f_rate = wb.add_format({"font_name": "Arial", "num_format": "0.0000", "border": 1})
    f_pct = wb.add_format({"font_name": "Arial", "num_format": "0.0%", "border": 1})
    f_int = wb.add_format({"font_name": "Arial", "num_format": "0", "border": 1})
    f_txt = wb.add_format({"font_name": "Arial", "border": 1})
    f_ok = wb.add_format({"bold": True, "font_name": "Arial", "font_color": "white",
                          "bg_color": _OK_BG, "border": 1, "align": "center"})
    f_breach = wb.add_format({"bold": True, "font_name": "Arial", "font_color": "white",
                              "bg_color": _BREACH_BG, "border": 1, "align": "center"})

    # ===================== Sheet 1: Summary =====================
    s = wb.add_worksheet("Summary")
    s.set_column("A:A", 30)
    s.set_column("B:B", 22)
    s.hide_gridlines(2)
    s.write("A1", "FX Book Risk Report", f_title)
    s.write("A2", f"Generated {datetime.now():%Y-%m-%d %H:%M} · "
                  "Demonstration tool — model prices on free/delayed data, not "
                  "investment advice.", f_sub)

    row = 3
    s.write(row, 0, "Book overview", f_hdr); s.write(row, 1, "", f_hdr); row += 1
    s.write(row, 0, "Book notional (USD)", f_label)
    s.write_number(row, 1, summary.get("notional", 0), f_num); row += 1
    s.write(row, 0, "Book value / MtM (USD)", f_label)
    s.write_number(row, 1, summary.get("book_value", 0), f_num); row += 1
    s.write(row, 0, "Confidence level", f_label)
    s.write_number(row, 1, summary.get("confidence", 0.99), f_pct); row += 2

    s.write(row, 0, "Value at Risk — 1 day (USD)", f_hdr)
    s.write(row, 1, "", f_hdr); row += 1
    for key, lbl in [("var_parametric", "Parametric"), ("var_historical", "Historical"),
                     ("var_montecarlo", "Monte Carlo"), ("var_ewma", "EWMA"),
                     ("var_student_t", "Student-t"), ("expected_shortfall", "Expected Shortfall")]:
        if key in summary:
            s.write(row, 0, lbl, f_label)
            s.write_number(row, 1, summary[key], f_num); row += 1
    row += 1

    # Limits block with colour-coded status.
    limits = summary.get("limits", [])
    if limits:
        s.write(row, 0, "Risk limits", f_hdr); s.write(row, 1, "Status", f_hdr); row += 1
        for lim in limits:
            s.write(row, 0, lim["label"], f_label)
            s.write(row, 1, lim["status"], f_ok if lim["ok"] else f_breach); row += 1
        row += 1

    # Footer: data source, disclaimer, and which VaR governs the limit check
    # above -- so the report is self-explanatory without the app alongside it.
    s.write(row, 0, "Data: yfinance (spot), FRED & ECB (rate curves), "
                    "GARCH(1,1)-t (volatility). Educational/demonstration tool "
                    "— not investment advice.", f_sub); row += 1
    s.write(row, 0, "The VaR limit above is checked against the GOVERNING VaR "
                    "= max(historical, Student-t) — the more conservative of "
                    "the two fat-tail-aware methods listed under 'Value at "
                    "Risk' — not any single figure in isolation.", f_sub)

    # ===================== Sheet 2: Positions =====================
    p = wb.add_worksheet("Positions")
    p.hide_gridlines(2)
    cols = ["Pair", "Side", "Notional", "Rate quoted", "Tenor (days)", "MtM (USD)"]
    # Widths sized to content: "Provider sells EUR" (18 chars) is the longest
    # Side value; the rest are sized to their formatted numbers/headers.
    widths = [10, 20, 14, 13, 13, 14]
    for i, w in enumerate(widths):
        p.set_column(i, i, w)
    p.write(0, 0, "Positions", f_title)
    hdr_row = 2
    for i, c in enumerate(cols):
        p.write(hdr_row, i, c, f_hdr)
    for r, pos in enumerate(book_rows, start=hdr_row + 1):
        p.write(r, 0, pos.get("pair", ""), f_txt)
        p.write(r, 1, pos.get("side", ""), f_txt)
        p.write_number(r, 2, pos.get("notional", 0), f_num)
        p.write_number(r, 3, pos.get("rate", 0), f_rate)
        p.write_number(r, 4, pos.get("tenor", 0), f_int)
        p.write_number(r, 5, pos.get("mtm", 0), f_num)

    # ===================== Sheet 3: Risk detail =====================
    d = wb.add_worksheet("Risk detail")
    d.hide_gridlines(2)
    # A:A sized for the longest section header, "Key-rate DV01 by tenor (USD
    # per 1bp)" (36 chars), so it never gets visually truncated.
    d.set_column("A:A", 38); d.set_column("B:B", 16)
    d.write(0, 0, "Risk detail", f_title)
    rr = 2
    d.write(rr, 0, "DV01 by currency (USD per 1bp)", f_hdr)
    d.write(rr, 1, "", f_hdr); rr += 1
    for ccy, dv in risk_detail.get("dv01_by_ccy", {}).items():
        d.write(rr, 0, ccy, f_label); d.write_number(rr, 1, dv, f_num2); rr += 1
    rr += 1
    d.write(rr, 0, "Key-rate DV01 by tenor (USD per 1bp)", f_hdr)
    d.write(rr, 1, "", f_hdr); rr += 1
    for bucket, dv in risk_detail.get("dv01_by_tenor", {}).items():
        d.write(rr, 0, bucket, f_label); d.write_number(rr, 1, dv, f_num2); rr += 1
    rr += 1
    if "liquidity" in risk_detail:
        d.write(rr, 0, "Liquidity buffer (USD)", f_label)
        d.write_number(rr, 1, risk_detail["liquidity"], f_num); rr += 2
    stress = risk_detail.get("stress", {})
    if stress:
        d.write(rr, 0, "Stress scenarios — P&L (USD)", f_hdr)
        d.write(rr, 1, "", f_hdr); rr += 1
        for name, pnl in stress.items():
            d.write(rr, 0, name, f_label); d.write_number(rr, 1, pnl, f_num); rr += 1

    wb.close()
    buffer.seek(0)
    return buffer.getvalue()

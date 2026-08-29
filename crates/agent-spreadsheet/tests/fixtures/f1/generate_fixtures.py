#!/usr/bin/env python3
"""Generate adversarial OOXML fixtures without third-party Python packages."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>
"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""

WORKBOOK = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
  <calcPr calcId="0" calcMode="manual" fullCalcOnLoad="0" forceFullCalc="0"/>
</workbook>
"""

WORKBOOK_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""

STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>
"""


def number(ref, value):
    return f'<c r="{ref}" t="n"><v>{value}</v></c>'


def text(ref, value):
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'


def error(ref, value, formula=None):
    f = f"<f>{escape(formula)}</f>" if formula is not None else ""
    return f'<c r="{ref}" t="e">{f}<v>{escape(value)}</v></c>'


def formula(ref, expression, cached=None, cell_type=None):
    attrs = f' r="{ref}"'
    if cell_type:
        attrs += f' t="{cell_type}"'
    value = "" if cached is None else f"<v>{escape(str(cached))}</v>"
    return f"<c{attrs}><f>{escape(expression)}</f>{value}</c>"


def empty_string_formula(ref, expression):
    return f'<c r="{ref}" t="str"><f>{escape(expression)}</f><v></v></c>'


def write_book(name, rows):
    row_xml = []
    for row_num, cells in rows:
        row_xml.append(f'<row r="{row_num}">{"".join(cells)}</row>')
    sheet = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>{"".join(row_xml)}</sheetData>
</worksheet>
'''
    path = FIXTURES / name
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", ROOT_RELS)
        zf.writestr("xl/workbook.xml", WORKBOOK)
        zf.writestr("xl/_rels/workbook.xml.rels", WORKBOOK_RELS)
        zf.writestr("xl/styles.xml", STYLES)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)
    return path


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for old in FIXTURES.glob("*.xlsx"):
        old.unlink()

    write_book("baseline.xlsx", [(1, [number("A1", 1), number("B1", 2), number("C1", 3)])])
    write_book("unevaluated_broken.xlsx", [(1, [number("A1", 1), formula("B1", "UNKNOWNFN(A1)"), formula("C1", "1/0")])])
    write_book("partial.xlsx", [(1, [number("A1", 1), formula("B1", "A1+1", 2), formula("C1", "1/0"), formula("D1", "B1+1", 3)])])
    write_book("stale_cache.xlsx", [(1, [number("A1", 2), formula("B1", "A1*10", 10)])])
    write_book("real_errors.xlsx", [(1, [error("A1", "#DIV/0!", "1/0")]), (2, [error("A2", "#NAME?", "UNKNOWNFN(1)")]), (3, [error("A3", "#REF!")])])
    write_book("ratio_600_percent.xlsx", [(1, [text("A1", "anchor"), *[formula(f"{c}1", "1+1") for c in "BCDEFG"]])])
    write_book("all_formulas_uncached.xlsx", [(1, [formula("A1", "1+1"), formula("B1", "A1+1")])])
    write_book("evaluated_empty_string.xlsx", [(1, [empty_string_formula("A1", 'IF(1=1,"","x")')])])
    print("\n".join(str(p) for p in sorted(FIXTURES.glob("*.xlsx"))))


if __name__ == "__main__":
    main()

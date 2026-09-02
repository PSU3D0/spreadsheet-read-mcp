# Ticket: 5007 Drawing Parts From openpyxl-Authored Workbooks Are Invisible

## Depends On
- none (found during 5004)

## Why
umya matches drawing and chart XML by literal prefixed names (`xdr:wsDr`, `c:chartSpace`) and expects relative relationship targets. openpyxl writes default-namespaced roots and absolute `Target="/xl/..."` paths. umya therefore reports no drawings for such workbooks, so the renderer cannot even raise `chart_omitted` or `image_omitted`, and every other surface that inspects drawings is blind to them. The 5004 fixture generator rewrites both forms to what Excel writes as a workaround.

## Scope
- Reproduce with an unmodified openpyxl chart and image workbook.
- Fix in the PSU3D0 umya fork (namespace-aware matching, absolute target normalization) and bump the workspace patch rev.
- Remove the rewrite step from `crates/agent-spreadsheet-render/tests/fixtures/generate.py` and confirm the chart and image warnings still fire.

## Definition of Done
- openpyxl-authored charts and images produce the structured warnings without fixture rewriting.

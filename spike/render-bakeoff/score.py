#!/usr/bin/env python3
"""Scoring for the ticket 5003 bake-off. Run with:
    uv run --with pillow --with numpy python3 score.py

Two independent scores, because they measure different things.

1. Shape score against an oracle. Neither oracle produces a registered image:
   LibreOffice `--convert-to png` and office2pdf both paginate to a print page
   with no row/column headings and their own fit-to-width, while the candidates
   render a bounded A1:M40 grid window. A per-pixel diff between them is
   meaningless. What survives the mismatch, after cropping to the inked region
   exactly as `crop_png_in_place` in screenshot.rs does, is the *shape* of the
   ink: aspect ratio and ink density. Reported as a weak signal only.

2. Registered pixel diff between spike-D variants, where geometry is identical
   and only the text stack or the font subset changed. This one is exact.

Plus a per-workbook structural checklist read straight out of the xlsx.
"""
import json, os, sys, zipfile
from collections import Counter
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
Image.MAX_IMAGE_PIXELS = None


def luma(path):
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.uint8)


def ink_stats(a, thresh=250):
    mask = a < thresh
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    return {"box": [x0, y0, w, h], "ink": int(mask.sum()), "density": float(mask.sum()) / (w * h)}


def shape_score(cand, oracle):
    try:
        c, o = ink_stats(luma(cand)), ink_stats(luma(oracle))
    except Exception as e:
        return {"error": str(e)[:80]}
    if not c or not o:
        return {"error": "blank"}
    car = c["box"][2] / max(c["box"][3], 1)
    oar = o["box"][2] / max(o["box"][3], 1)
    return {
        "cand_box": c["box"], "oracle_box": o["box"],
        "aspect_score": round(min(car, oar) / max(car, oar), 3),
        "density_cand": round(c["density"], 4),
        "density_oracle": round(o["density"], 4),
        "density_score": round(min(c["density"], o["density"]) / max(c["density"], o["density"]), 3),
    }


def pixel_diff(a_path, b_path):
    a, b = luma(a_path), luma(b_path)
    if a.shape != b.shape:
        return {"same_dims": False, "a": list(a.shape), "b": list(b.shape)}
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    n = d.size
    return {
        "same_dims": True, "pixels": int(n),
        "differing_pct": round(100.0 * int((d > 0).sum()) / n, 4),
        "differing_gt16_pct": round(100.0 * int((d > 16).sum()) / n, 4),
        "max_delta": int(d.max()),
    }


def checklist(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    st = z.read("xl/styles.xml").decode("utf8", "replace") if "xl/styles.xml" in names else ""
    sheets = sorted(n for n in names if n.startswith("xl/worksheets/sheet"))
    sh = z.read(sheets[0]).decode("utf8", "replace") if sheets else ""
    return {
        "values": ("<v>" in sh) or ("<is>" in sh),
        "col_widths": "<col " in sh and "width=" in sh,
        "row_heights": "customHeight" in sh or " ht=" in sh,
        "merges": "<mergeCell " in sh,
        "fills": st.count("<patternFill") > 2,
        "borders": st.count("<border>") > 1,
        "alignment": "<alignment" in st,
        "wrap": "wrapText" in st,
        "number_formats": "<numFmt " in st,
        "hidden_rows": 'hidden="1"' in sh and "<row " in sh,
        "hidden_cols": 'hidden="1"' in sh and "<col " in sh,
        "gridlines_off": 'showGridLines="0"' in sh,
        "conditional_format": "<conditionalFormatting" in sh,
        "images": any(n.startswith("xl/media/") for n in names),
        "charts": any(n.startswith("xl/charts/") for n in names),
    }


def main():
    corpus = os.path.join(HERE, "corpus")
    files = {}
    for group in sorted(os.listdir(corpus)):
        gp = os.path.join(corpus, group)
        if not os.path.isdir(gp):
            continue
        for f in sorted(os.listdir(gp)):
            if f.endswith(".xlsx"):
                files[f"{group}__{f[:-5]}"] = os.path.join(gp, f)

    lo = os.path.join(HERE, "out", "oracle-libreoffice")
    o2p = os.path.join(HERE, "out", "ref-c-png")
    runs = {
        "ref-a": os.path.join(HERE, "out", "run1", "ref-a"),
        "ref-b": os.path.join(HERE, "out", "run1", "ref-b"),
        "spike-d": os.path.join(HERE, "out", "run-full", "spike-d"),
    }

    out = []
    for name, xlsx in files.items():
        rec = {"file": name, "checklist": checklist(xlsx)}
        for oname, odir in (("vs_libreoffice", lo), ("vs_office2pdf", o2p)):
            op = os.path.join(odir, f"{name}.png")
            rec[oname] = {}
            for backend, d in runs.items():
                p = os.path.join(d, f"{name}.png")
                rec[oname][backend] = (
                    shape_score(p, op) if os.path.exists(p) and os.path.exists(op)
                    else {"error": "missing"}
                )
        full = os.path.join(HERE, "out", "run-full", "spike-d", f"{name}.png")
        for label, other in (
            ("subset_vs_full", os.path.join(HERE, "out", "run-subset", "spike-d", f"{name}.png")),
            ("abglyph_vs_rustybuzz", os.path.join(HERE, "out", "run-abglyph", "spike-d", f"{name}.png")),
        ):
            if os.path.exists(full) and os.path.exists(other):
                rec[label] = pixel_diff(other, full)
        out.append(rec)
        print(json.dumps(rec))

    agg = Counter()
    for r in out:
        for k, v in r["checklist"].items():
            if v:
                agg[k] += 1
    sys.stderr.write("checklist coverage: " + json.dumps(dict(agg)) + "\n")


if __name__ == "__main__":
    main()

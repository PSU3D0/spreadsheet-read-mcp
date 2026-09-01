#!/usr/bin/env bash
# Generates styled fixtures for the renderer bake-off using the asp CLI.
# Row heights and hidden rows/columns are NOT expressible through any asp write
# surface as of 0.14.0; those two fixtures are produced with openpyxl instead.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ASP="${ASP:-$HERE/../../../target/debug/asp}"
OUT="$HERE/corpus/generated"
TMP="${TMPDIR:-/tmp}/genfix"
mkdir -p "$OUT" "$TMP"
rm -f "$OUT"/gen_*.xlsx

new() { "$ASP" workbook create "$OUT/$1.xlsx" --sheets Sheet1 --overwrite >/dev/null; }
cells() { "$ASP" write cells "$OUT/$1.xlsx" Sheet1 --edits-file "$TMP/$1.edits.txt" --in-place --formula-parse-policy warn >/dev/null; }
style() { "$ASP" write batch style "$OUT/$1.xlsx" --ops "@$TMP/$1.style.json" --in-place >/dev/null; }
colsz() { "$ASP" write batch column-size "$OUT/$1.xlsx" --ops "@$TMP/$1.cols.json" --in-place >/dev/null; }
struct() { "$ASP" write batch structure "$OUT/$1.xlsx" --ops "@$TMP/$1.struct.json" --in-place >/dev/null; }
layout() { "$ASP" write batch sheet-layout "$OUT/$1.xlsx" --ops "@$TMP/$1.layout.json" --in-place >/dev/null; }

python3 - "$TMP" <<'PY'
import json, sys, os
tmp = sys.argv[1]
os.makedirs(tmp, exist_ok=True)
def w(name, kind, obj):
    with open(os.path.join(tmp, f"{name}.{kind}.json"), "w") as f:
        json.dump(obj, f)
def wcells(name, cellmap):
    """cellmap: dict addr -> value. Strings starting with '=' become formulas."""
    lines=[]
    for k,v in cellmap.items():
        if isinstance(v,str) and v.startswith("="):
            lines.append(f"{k}={v}")       # '=' + '=FORMULA' -> '==FORMULA'
        elif isinstance(v,bool):
            lines.append(f"{k}={'TRUE' if v else 'FALSE'}")
        else:
            lines.append(f"{k}={v}")
    with open(os.path.join(tmp, f"{name}.edits.txt"), "w") as f:
        f.write("\n".join(lines)+"\n")
def col(i):
    s=""
    i+=1
    while i: i,r = divmod(i-1,26); s = chr(65+r)+s
    return s

# ---- gen_01_fonts
fonts = ["Arial","Times New Roman","Courier New","Georgia","Verdana","Trebuchet MS","Calibri"]
cs = {"A1": "Font sample matrix"}
sops = [{"sheet_name":"Sheet1","target":{"kind":"range","range":"A1:A1"},
         "patch":{"font":{"bold":True,"size":16.0,"name":"Georgia","color":"FF1F3864"}}}]
for r,fn in enumerate(fonts, start=3):
    cs[f"A{r}"]=fn; cs[f"B{r}"]="AgWy 0123 ,.;:"; cs[f"C{r}"]="Bold"; cs[f"D{r}"]="Italic"; cs[f"E{r}"]="BoldItalic"; cs[f"F{r}"]="Underline"; cs[f"G{r}"]="Strike"
    sops += [
      {"sheet_name":"Sheet1","target":{"kind":"range","range":f"B{r}:B{r}"},"patch":{"font":{"name":fn,"size":11.0}}},
      {"sheet_name":"Sheet1","target":{"kind":"range","range":f"C{r}:C{r}"},"patch":{"font":{"name":fn,"bold":True}}},
      {"sheet_name":"Sheet1","target":{"kind":"range","range":f"D{r}:D{r}"},"patch":{"font":{"name":fn,"italic":True}}},
      {"sheet_name":"Sheet1","target":{"kind":"range","range":f"E{r}:E{r}"},"patch":{"font":{"name":fn,"bold":True,"italic":True}}},
      {"sheet_name":"Sheet1","target":{"kind":"range","range":f"F{r}:F{r}"},"patch":{"font":{"name":fn,"underline":"single"}}},
      {"sheet_name":"Sheet1","target":{"kind":"range","range":f"G{r}:G{r}"},"patch":{"font":{"name":fn,"strikethrough":True}}},
    ]
for r,sz in enumerate([6.0,8.0,11.0,14.0,20.0,28.0], start=12):
    cs[f"A{r}"]=f"{sz:g}pt"; cs[f"B{r}"]="Size sample"
    sops.append({"sheet_name":"Sheet1","target":{"kind":"range","range":f"B{r}:B{r}"},"patch":{"font":{"size":sz}}})
wcells("gen_01_fonts", cs)
w("gen_01_fonts","style",{"ops":sops})
w("gen_01_fonts","cols",{"sheet_name":"Sheet1","ops":[{"target":{"kind":"columns","range":"A:A"},"size":{"kind":"width","width_chars":18.0}},{"target":{"kind":"columns","range":"B:B"},"size":{"kind":"width","width_chars":22.0}}]})

# ---- gen_02_fills
palette = ["FFFF0000","FF00B050","FF0070C0","FFFFC000","FF7030A0","FFFFFF00","FF00B0F0","FF002060","FFC00000","FFE2EFDA","FFD9E1F2","FFFCE4D6"]
cs = {"A1":"Solid fills"}
sops=[]
for i,c in enumerate(palette):
    r = 3 + i//4; cc = col(i%4)
    cs[f"{cc}{r}"]=c[2:]
    sops.append({"sheet_name":"Sheet1","target":{"kind":"range","range":f"{cc}{r}:{cc}{r}"},
                 "patch":{"fill":{"kind":"pattern","pattern_type":"solid","foreground_color":c},
                          "font":{"color":"FFFFFFFF" if i%2 else "FF000000","bold":True}}})
for i,pt in enumerate(["gray125","lightGray","darkGrid","lightUp","mediumGray"]):
    r=8+i; cs[f"A{r}"]=pt
    sops.append({"sheet_name":"Sheet1","target":{"kind":"range","range":f"B{r}:C{r}"},
                 "patch":{"fill":{"kind":"pattern","pattern_type":pt,"foreground_color":"FF4472C4","background_color":"FFFFFFFF"}}})
wcells("gen_02_fills", cs)
w("gen_02_fills","style",{"ops":sops})

# ---- gen_03_borders
styles=["thin","medium","thick","double","dashed","dotted","dashDot","dashDotDot","hair","mediumDashed","slantDashDot","mediumDashDot"]
cs={"A1":"Border styles"}; sops=[]
for i,st in enumerate(styles):
    r=3+i*2; cs[f"A{r}"]=st
    side={"style":st,"color":"FF000000"}
    sops.append({"sheet_name":"Sheet1","target":{"kind":"range","range":f"C{r}:D{r}"},
                 "patch":{"borders":{"top":side,"bottom":side,"left":side,"right":side}}})
sops.append({"sheet_name":"Sheet1","target":{"kind":"range","range":"F3:H10"},
             "patch":{"borders":{"top":{"style":"thick","color":"FFC00000"},
                                 "bottom":{"style":"thick","color":"FFC00000"},
                                 "left":{"style":"medium","color":"FF0070C0"},
                                 "right":{"style":"medium","color":"FF0070C0"},
                                 "horizontal":{"style":"hair","color":"FF808080"},
                                 "vertical":{"style":"dotted","color":"FF808080"}}}})
wcells("gen_03_borders", cs)
w("gen_03_borders","style",{"ops":sops})

# ---- gen_04_merges
cs={"A1":"Quarterly Report (merged title)","A3":"Region","B3":"Q1","D3":"Q2","F3":"Total"}
rows=[("North",100,110,120,130,460),("South",90,95,105,115,405),("East",80,85,95,100,360),("West",70,75,85,90,320)]
for i,(n,a,b,c,d,t) in enumerate(rows):
    r=5+i; cs[f"A{r}"]=n; cs[f"B{r}"]=a; cs[f"C{r}"]=b; cs[f"D{r}"]=c; cs[f"E{r}"]=d; cs[f"F{r}"]=t
wcells("gen_04_merges", cs)
w("gen_04_merges","struct",{"ops":[
  {"kind":"merge_cells","sheet_name":"Sheet1","target_range":"A1:F1"},
  {"kind":"merge_cells","sheet_name":"Sheet1","target_range":"B3:C3"},
  {"kind":"merge_cells","sheet_name":"Sheet1","target_range":"D3:E3"},
  {"kind":"merge_cells","sheet_name":"Sheet1","target_range":"F3:F4"},
  {"kind":"merge_cells","sheet_name":"Sheet1","target_range":"A3:A4"},
]})
w("gen_04_merges","style",{"ops":[
  {"sheet_name":"Sheet1","target":{"kind":"range","range":"A1:F1"},
   "patch":{"font":{"bold":True,"size":16.0,"color":"FFFFFFFF"},"fill":{"kind":"pattern","pattern_type":"solid","foreground_color":"FF1F3864"},
            "alignment":{"horizontal":"center","vertical":"center"}}},
  {"sheet_name":"Sheet1","target":{"kind":"range","range":"A3:F4"},
   "patch":{"font":{"bold":True},"fill":{"kind":"pattern","pattern_type":"solid","foreground_color":"FFD9E1F2"},
            "alignment":{"horizontal":"center","vertical":"center"},
            "borders":{"top":{"style":"thin"},"bottom":{"style":"thin"},"left":{"style":"thin"},"right":{"style":"thin"}}}},
]})

# ---- gen_05_numfmt
fmts=[("0","1234.5678"),("0.00","1234.5678"),("#,##0","1234567"),("#,##0.00","1234567.891"),
      ("0%","0.4567"),("0.00%","0.4567"),("0.00E+00","1234.5678"),
      ('"$"#,##0.00',"1234.5"),('"$"#,##0.00_);[Red]("$"#,##0.00)',"-1234.5"),
      ("yyyy-mm-dd","45000"),("dd/mm/yyyy","45000"),("mmm d, yyyy","45000"),("h:mm:ss AM/PM","0.53125"),
      ("yyyy-mm-dd hh:mm","45000.53125"),("# ?/?","2.25"),("@","hello"),
      ("[$-409]d\\-mmm\\-yy;@","45000"),('_(* #,##0.00_);_(* \\(#,##0.00\\);_(* "-"??_);_(@_)',"-9876.54")]
cs={"A1":"Number formats"}; sops=[]
for i,(f,v) in enumerate(fmts):
    r=3+i; cs[f"A{r}"]=f; cs[f"B{r}"]=float(v) if v.replace('.','').replace('-','').isdigit() else v
    sops.append({"sheet_name":"Sheet1","target":{"kind":"range","range":f"B{r}:B{r}"},"patch":{"number_format":f}})
wcells("gen_05_numfmt", cs)
w("gen_05_numfmt","style",{"ops":sops})
w("gen_05_numfmt","cols",{"sheet_name":"Sheet1","ops":[{"target":{"kind":"columns","range":"A:A"},"size":{"kind":"width","width_chars":38.0}},{"target":{"kind":"columns","range":"B:B"},"size":{"kind":"width","width_chars":24.0}}]})

# ---- gen_06_align_wrap
h=["left","center","right","justify","fill","distributed"]
v=["top","center","bottom"]
cs={"A1":"Alignment and wrap"}; sops=[]
for ci,hh in enumerate(h):
    cs[f"{col(ci+1)}2"]=hh
    for ri,vv in enumerate(v):
        r=3+ri
        cs[f"A{r}"]=vv
        cs[f"{col(ci+1)}{r}"]="Ax"
        sops.append({"sheet_name":"Sheet1","target":{"kind":"range","range":f"{col(ci+1)}{r}:{col(ci+1)}{r}"},
                     "patch":{"alignment":{"horizontal":hh,"vertical":vv},
                              "borders":{"top":{"style":"hair"},"bottom":{"style":"hair"},"left":{"style":"hair"},"right":{"style":"hair"}}}})
cs["A8"]="wrapped"
cs["B8"]="This is a deliberately long sentence that must wrap onto several lines inside one merged-free cell."
sops.append({"sheet_name":"Sheet1","target":{"kind":"range","range":"B8:B8"},"patch":{"alignment":{"wrap_text":True,"vertical":"top"}}})
cs["A10"]="rotated"
for i,deg in enumerate([0,15,30,45,60,90]):
    cs[f"{col(i+1)}10"]=f"rot{deg}"
    sops.append({"sheet_name":"Sheet1","target":{"kind":"range","range":f"{col(i+1)}10:{col(i+1)}10"},"patch":{"alignment":{"text_rotation":deg}}})
cs["A12"]="overflow"
cs["B12"]="Unclipped text that should spill across the empty neighbouring cells to the right."
wcells("gen_06_align_wrap", cs)
w("gen_06_align_wrap","style",{"ops":sops})
w("gen_06_align_wrap","cols",{"sheet_name":"Sheet1","ops":[{"target":{"kind":"columns","range":"B:B"},"size":{"kind":"width","width_chars":28.0}}]})

# ---- gen_07_colwidths
cs={"A1":"Column widths"}
widths=[2.0,4.5,8.43,12.0,18.0,25.0,40.0,60.0]
for i,wd in enumerate(widths):
    c=col(i)
    cs[f"{c}2"]=f"{wd:g}"
    cs[f"{c}3"]="MMMMMMMMMMMM"
    cs[f"{c}4"]=12345.678
wcells("gen_07_colwidths", cs)
w("gen_07_colwidths","cols",{"sheet_name":"Sheet1","ops":[
   {"target":{"kind":"columns","range":f"{col(i)}:{col(i)}"},"size":{"kind":"width","width_chars":wd}} for i,wd in enumerate(widths)]})

# ---- gen_10_gridlines
cs={"A1":"Gridlines off + frozen panes"}
for r in range(3,20):
    for c in range(0,6):
        cs[f"{col(c)}{r}"] = r*10+c
wcells("gen_10_gridlines", cs)
w("gen_10_gridlines","layout",{"ops":[
  {"kind":"set_gridlines","sheet_name":"Sheet1","show":False},
  {"kind":"freeze_panes","sheet_name":"Sheet1","freeze_rows":2,"freeze_cols":1},
]})

# ---- gen_11_dashboard (everything at once)
cs={"A1":"ACME Corp — FY2026 Operating Summary"}
hdr=["Segment","Revenue","COGS","Gross Margin","Margin %","YoY"]
for i,hn in enumerate(hdr): cs[f"{col(i)}3"]=hn
data=[("Cloud Platform",12450000,5320000,7130000,0.5727,0.184),
      ("Professional Services",4210000,3110000,1100000,0.2613,-0.032),
      ("Hardware",8900000,7450000,1450000,0.1629,0.071),
      ("Support & Maintenance",3300000,900000,2400000,0.7273,0.115),
      ("Other",150000,60000,90000,0.6000,0.000)]
for i,row in enumerate(data):
    r=4+i
    for j,val in enumerate(row): cs[f"{col(j)}{r}"]=val
cs["A9"]="Total"; cs["B9"]="=SUM(B4:B8)"; cs["C9"]="=SUM(C4:C8)"; cs["D9"]="=SUM(D4:D8)"; cs["E9"]="=D9/B9"
cs["A11"]="Note"
cs["B11"]="Margins are unaudited; hardware includes a one-off inventory write-down of $1.2M recognised in Q3."
wcells("gen_11_dashboard", cs)
money='"$"#,##0'
w("gen_11_dashboard","style",{"ops":[
 {"sheet_name":"Sheet1","target":{"kind":"range","range":"A1:F1"},"patch":{"font":{"bold":True,"size":18.0,"name":"Georgia","color":"FFFFFFFF"},"fill":{"kind":"pattern","pattern_type":"solid","foreground_color":"FF1F3864"},"alignment":{"horizontal":"center","vertical":"center"}}},
 {"sheet_name":"Sheet1","target":{"kind":"range","range":"A3:F3"},"patch":{"font":{"bold":True,"color":"FFFFFFFF"},"fill":{"kind":"pattern","pattern_type":"solid","foreground_color":"FF4472C4"},"alignment":{"horizontal":"center"},"borders":{"bottom":{"style":"medium","color":"FF1F3864"}}}},
 {"sheet_name":"Sheet1","target":{"kind":"range","range":"B4:D9"},"patch":{"number_format":money,"alignment":{"horizontal":"right"}}},
 {"sheet_name":"Sheet1","target":{"kind":"range","range":"E4:F9"},"patch":{"number_format":'0.0%;[Red]-0.0%',"alignment":{"horizontal":"right"}}},
 {"sheet_name":"Sheet1","target":{"kind":"range","range":"A9:F9"},"patch":{"font":{"bold":True},"fill":{"kind":"pattern","pattern_type":"solid","foreground_color":"FFD9E1F2"},"borders":{"top":{"style":"double","color":"FF1F3864"}}}},
 {"sheet_name":"Sheet1","target":{"kind":"range","range":"A4:A8"},"patch":{"font":{"name":"Verdana","size":10.0}}},
 {"sheet_name":"Sheet1","target":{"kind":"range","range":"B11:F11"},"patch":{"alignment":{"wrap_text":True,"vertical":"top"},"font":{"italic":True,"size":9.0,"color":"FF595959"}}},
]})
w("gen_11_dashboard","cols",{"sheet_name":"Sheet1","ops":[
 {"target":{"kind":"columns","range":"A:A"},"size":{"kind":"width","width_chars":26.0}},
 {"target":{"kind":"columns","range":"B:D"},"size":{"kind":"width","width_chars":16.0}},
 {"target":{"kind":"columns","range":"E:F"},"size":{"kind":"width","width_chars":11.0}}]})
w("gen_11_dashboard","struct",{"ops":[
 {"kind":"merge_cells","sheet_name":"Sheet1","target_range":"A1:F1"},
 {"kind":"merge_cells","sheet_name":"Sheet1","target_range":"B11:F11"}]})

# ---- gen_12_unicode
cs={"A1":"Unicode, currency and punctuation"}
samples=["Ärger Ünïcödé","naïve café","£1,234.56","€9.876,54","¥120,000","₹45,000","«guillemets»",
         "em—dash, en–dash","“smart quotes”","±3.5 °C","½ ¼ ¾","→ ← ↑ ↓","α β γ Ω","ß æ ø å","N.º 12","© 2026 ACME™"]
for i,s in enumerate(samples): cs[f"A{3+i}"]=s; cs[f"B{3+i}"]=len(s)
wcells("gen_12_unicode", cs)
w("gen_12_unicode","cols",{"sheet_name":"Sheet1","ops":[{"target":{"kind":"columns","range":"A:A"},"size":{"kind":"width","width_chars":30.0}}]})
PY

for n in gen_01_fonts gen_02_fills gen_03_borders gen_04_merges gen_05_numfmt gen_06_align_wrap gen_07_colwidths gen_10_gridlines gen_11_dashboard gen_12_unicode; do
  new "$n"
  [ -f "$TMP/$n.edits.txt" ]   && cells "$n"
  [ -f "$TMP/$n.struct.json" ] && struct "$n"
  [ -f "$TMP/$n.style.json" ]  && style "$n"
  [ -f "$TMP/$n.cols.json" ]   && colsz "$n"
  [ -f "$TMP/$n.layout.json" ] && layout "$n"
  echo "ok $n"
done

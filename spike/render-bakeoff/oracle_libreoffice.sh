#!/usr/bin/env bash
# Reference C oracle: LibreOffice headless PNG per corpus workbook.
# Uses --convert-to png (page 1 only), which is the fallback path already present
# in crates/agent-spreadsheet/src/recalc/screenshot.rs when the macro is unavailable.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/out/oracle-libreoffice"
PROF="$HERE/out/loprofile"   # soffice needs a writable profile AND a writable /tmp for its named pipe
mkdir -p "$OUT" "$PROF"
rm -rf "$OUT"/*.png
: > "$OUT/timings.tsv"
printf 'file\tstatus\tms\tbytes\twidth\theight\n' >> "$OUT/timings.tsv"
find "$HERE/corpus" -name '*.xlsx' | sort | while read -r f; do
  name="$(basename "$(dirname "$f")")__$(basename "$f" .xlsx)"
  start=$(date +%s%N)
  timeout 120 /bin/soffice --headless --norestore \
      "-env:UserInstallation=file://$PROF" \
      --convert-to png --outdir "$OUT/$name" "$f" >/dev/null 2>&1
  rc=$?
  end=$(date +%s%N); ms=$(( (end-start)/1000000 ))
  png="$(find "$OUT/$name" -name '*.png' 2>/dev/null | head -1)"
  if [ -n "$png" ]; then
    mv "$png" "$OUT/$name.png"; rm -rf "$OUT/$name"
    sz=$(stat -c%s "$OUT/$name.png")
    dim=$(python3 -c "import struct,sys;d=open(sys.argv[1],'rb').read(33);print('%d\t%d'%struct.unpack('>II',d[16:24]))" "$OUT/$name.png")
    printf '%s\tok\t%s\t%s\t%s\n' "$name" "$ms" "$sz" "$dim" >> "$OUT/timings.tsv"
  else
    rm -rf "$OUT/$name"
    printf '%s\tfail_rc%s\t%s\t0\t0\t0\n' "$name" "$rc" "$ms" >> "$OUT/timings.tsv"
  fi
done
echo "LIBREOFFICE ORACLE DONE"

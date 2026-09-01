#!/usr/bin/env bash
# wasm32 size deltas for the text stack and font strategy.
# Release profile is set in the spike workspace root: opt-level=z, lto=true,
# codegen-units=1, panic=abort, strip=true.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
CARGO="${CARGO_BIN:-cargo}"
TC="${TOOLCHAIN:-+1.97.1}"
out="$HERE/out/wasm-sizes.tsv"
printf 'variant\tbytes\tgz_bytes\n' > "$out"
run() {
  local name="$1"; shift
  $CARGO $TC build --release -p text-wasm --target wasm32-unknown-unknown \
      --no-default-features --features "$1" --config 'build.rustc-wrapper=""' >/dev/null 2>&1
  local f="$HERE/target/wasm32-unknown-unknown/release/text_wasm.wasm"
  if [ -f "$f" ]; then
    local b gz
    b=$(stat -c%s "$f")
    gz=$(gzip -9 -c "$f" | wc -c)
    printf '%s\t%s\t%s\n' "$name" "$b" "$gz" >> "$out"
    cp "$f" "$HERE/out/wasm-$name.wasm"
  else
    printf '%s\tFAIL\tFAIL\n' "$name" >> "$out"
  fi
  rm -f "$f"
}
run baseline                 "baseline"
run rustybuzz-full           "text-rustybuzz,font-full"
run rustybuzz-subset         "text-rustybuzz,font-subset"
run abglyph-full             "text-abglyph,font-full"
run abglyph-subset           "text-abglyph,font-subset"
cat "$out"

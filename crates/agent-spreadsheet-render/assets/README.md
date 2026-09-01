# Embedded fonts

Two faces, subset, compiled into the crate with `include_bytes!`. No system
fonts are ever loaded, which is what makes PNG goldens reproducible across
machines and across native and `wasm32`.

| file | bytes | source |
|---|---|---|
| `Carlito-Regular-subset.ttf` | 70,580 | Carlito 1.103 Regular (628,032 bytes full) |
| `Carlito-Bold-subset.ttf` | 78,912 | Carlito 1.103 Bold (682,468 bytes full) |

149,492 bytes for the pair, 11.4% of the 1,310,500 bytes the full faces cost.

Carlito is metrically compatible with Calibri, which is what the ported Excel
column metric model assumes (`CALIBRI_DIGIT_ADVANCE_EM = 0.506836` applies to
both). Italic is synthesised by a 0.21 shear; there is no italic face.

## License

SIL Open Font License 1.1 — see `OFL.txt`. Copyright 2013 The Carlito Project
Authors (<https://github.com/googlefonts/carlito>), with Reserved Font Name
"Carlito". The subsets are modifications of the originals and remain under the
same license.

## Regenerating the subsets

The coverage is Latin-1 plus Latin Extended-A punctuation, currency (including
the euro and the rupee), arrows, maths signs and the fi/fl ligatures. Bump
`RENDERER_ID` in `src/lib.rs` and regenerate every PNG golden if this coverage
changes: it moves pixels.

```sh
UNI='U+0020-007E,U+00A0-00FF,U+0152-0153,U+0160-0161,U+0178,U+017D-017E,U+0192,U+02C6,U+02DC,U+2013-2014,U+2018-201A,U+201C-201E,U+2020-2022,U+2026,U+2030,U+2039-203A,U+2044,U+20AC,U+20B9,U+20BD,U+2122,U+2190-2193,U+2202,U+2206,U+220F,U+2211-2212,U+2215,U+221A,U+221E,U+222B,U+2248,U+2260,U+2264-2265,U+25CA,U+FB01-FB02'

uv run --with fonttools pyftsubset Carlito-Regular.ttf \
  --unicodes="$UNI" --layout-features='kern,liga,ccmp,locl' \
  --output-file=Carlito-Regular-subset.ttf

uv run --with fonttools pyftsubset Carlito-Bold.ttf \
  --unicodes="$UNI" --layout-features='kern,liga,ccmp,locl' \
  --output-file=Carlito-Bold-subset.ttf
```

## Missing glyphs

The 5003 bake-off measured the subset dropping glyphs *silently*: the Greek row
of the unicode fixture rendered blank. That must never ship. Any codepoint
outside the subset draws a visible `.notdef` box and raises the
`font_substituted` warning — see `src/text.rs` and
`tests/notdef.rs`.

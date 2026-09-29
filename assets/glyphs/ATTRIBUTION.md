# Glyph data provenance

The `*.json` files in this directory are **generated**. Do not edit them by hand:
they are rebuilt from upstream icon sources by `tools/build_glyphs.py`, which is
the only thing allowed to write here.

    python -m tools.build_glyphs            # regenerate
    python -m tools.build_glyphs --check    # fail if these files are stale

## Upstream

Geometry is derived from two icon sets, each pinned to a release. Which one a
glyph came from is recorded in its own `source` field — `"tabler:car@3.46.0"` or
`"lucide:robot-arm@1.47.0"` — so any file here can be traced back to the release
it was cut from, and `--check` can re-fetch exactly that.

| Set | Version | Licence | Glyphs here |
|---|---|---|---|
| [Tabler Icons](https://github.com/tabler/tabler-icons) | 3.46.0 | MIT | all but thirteen |
| [Lucide](https://github.com/lucide-icons/lucide) | 1.47.0 | ISC | thirteen |

Lucide is here for `robot-arm`, which Tabler does not have, and which is the only
drawing of an industrial arm in any of the icon sets this project surveyed. The
two sets are a good fit: both draw on a 24x24 grid, both are stroke-only at width
2 with round caps and joins, so they mix on one stage without a seam.

The pinned versions and download URLs live in `SETS` in `tools/build_glyphs.py`.
Bumping one is a deliberate act: the element counts below are asserted, so an
upstream reorder fails the build loudly rather than quietly drawing a different
picture.

Tabler's own spacer path is filtered out during the build. It is identified by
the `stroke="none"` attribute Tabler puts on it, rather than by matching its `d`
string — see `tools/build_glyphs.py`.

Lucide draws some shapes as elements other than `<path>` — a robot arm's base
joint is a `<circle>`, a cylinder is an `<ellipse>`, a clipboard board is a
`<rect>`. Those are converted to path data during the build, because the schema
stores `d` and a shape element the parser does not understand is a piece of the
picture that goes missing without saying so.

Two fields in each file come from upstream and one does not. `view_box` is the
set's design grid, copied as-is. `parts` is the upstream elements, regrouped so a
wheel can turn without the chassis turning. `ink_box` is **measured** from those
elements by `tools/path_bbox.py`, because the design grid is padded and fitting it
draws the glyph smaller than the body that asked for it.

## Licence

Tabler Icons is distributed under the MIT Licence, reproduced in full below as
the licence requires.

```
MIT License

Copyright (c) 2020-2026 Paweł Kuna

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Lucide is distributed under the ISC Licence, reproduced in full below — including
its own MIT section, because the licence file carries one for the subset of icons
it derived from the Feather project. None of the thirteen glyphs used here are in
that subset, but the file is reproduced whole rather than in part.

```
ISC License

Copyright (c) 2026 Lucide Icons and Contributors

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

---

The following Lucide icons are derived from the Feather project:

airplay, alert-circle, alert-octagon, alert-triangle, aperture, arrow-down-circle, arrow-down-left, arrow-down-right, arrow-down, arrow-left-circle, arrow-left, arrow-right-circle, arrow-right, arrow-up-circle, arrow-up-left, arrow-up-right, arrow-up, at-sign, calendar, cast, check, chevron-down, chevron-left, chevron-right, chevron-up, chevrons-down, chevrons-left, chevrons-right, chevrons-up, circle, clipboard, clock, code, columns, command, compass, corner-down-left, corner-down-right, corner-left-down, corner-left-up, corner-right-down, corner-right-up, corner-up-left, corner-up-right, crosshair, database, divide-circle, divide-square, dollar-sign, download, external-link, feather, frown, hash, headphones, help-circle, info, italic, key, layout, life-buoy, link-2, link, loader, lock, log-in, log-out, maximize, meh, minimize, minimize-2, minus-circle, minus-square, minus, monitor, moon, more-horizontal, more-vertical, move, music, navigation-2, navigation, octagon, pause-circle, percent, plus-circle, plus-square, plus, power, radio, rss, search, server, share, shopping-bag, sidebar, smartphone, smile, square, table-2, tablet, target, terminal, trash-2, trash, triangle, tv, type, upload, x-circle, x-octagon, x-square, x, zoom-in, zoom-out

The MIT License (MIT) (for the icons listed above)

Copyright (c) 2013-present Cole Bemis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

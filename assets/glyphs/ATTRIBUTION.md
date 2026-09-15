# Glyph data provenance

The `*.json` files in this directory are **generated**. Do not edit them by hand:
they are rebuilt from upstream icon sources by `tools/build_glyphs.py`, which is
the only thing allowed to write here.

    python -m tools.build_glyphs            # regenerate
    python -m tools.build_glyphs --check    # fail if these files are stale

## Upstream

Geometry is derived from **Tabler Icons** — <https://github.com/tabler/tabler-icons>,
MIT licensed — pinned to version **3.46.0**. The version is recorded in each
file's `source` field (`"tabler:car@3.46.0"`) so a glyph can be traced back to
the release it was cut from.

Tabler's own spacer path is filtered out during the build. It is identified by
the `stroke="none"` attribute Tabler puts on it, rather than by matching its `d`
string — see `tools/build_glyphs.py`.

Two fields in each file come from upstream and one does not. `view_box` is
Tabler's design grid, copied as-is. `parts` is Tabler's paths, regrouped so a
wheel can turn without the chassis turning. `ink_box` is **measured** from those
paths by `tools/path_bbox.py`, because the design grid is padded and fitting it
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

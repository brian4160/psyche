#!/usr/bin/env bash
# Render for-dad.md to a printable PDF (Letter, 11.5pt serif). Needs pandoc and a headless Chromium.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
pandoc for-dad.md -f markdown -t html5 -s --css paper.css --metadata title="Where This Is Going" -o build/for-dad.html
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("build/for-dad.html"); s = p.read_text()
s = re.sub(r'<header id="title-block-header">.*?</header>', '', s, flags=re.S)
s = s.replace('<table>', '<table style="table-layout: fixed; width: 100%">')
p.write_text(s)
PY
cp paper.css build/paper.css
CHROME="${CHROME:-/opt/pw-browsers/chromium}"
"$CHROME" --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer --print-to-pdf=build/for-dad.pdf "file://$PWD/build/for-dad.html" 2>/dev/null
cp build/for-dad.pdf for-dad.pdf
pdfinfo for-dad.pdf | grep Pages

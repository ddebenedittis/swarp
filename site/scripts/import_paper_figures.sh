#!/usr/bin/env bash
# Copy the paper's figures into src/assets/, so the page shows exactly what the paper prints.
#
#   site/scripts/import_paper_figures.sh /path/to/paper
#
# The paper repository is private, so its figures are copied rather than referenced. Plots are
# vectorized with pdftocairo (poppler-utils) and their width/height scaled up 3x: a column-width
# PDF figure is about 250 pt wide, which the page would otherwise render at its intrinsic ~330 px.
# The viewBox is untouched, so the SVG stays crisp at any size.
set -euo pipefail

paper=${1:?usage: $0 /path/to/paper}
assets=$(cd "$(dirname "$0")/../src/assets" && pwd)

mkdir -p "$assets/scenes"
cp "$paper"/img/scenes/*.png "$assets/scenes/"

for fig in ringroad_scaling ringroad_transfer; do
  pdftocairo -svg "$paper/img/$fig.pdf" "$assets/$fig.svg"
  python3 - "$assets/$fig.svg" <<'EOF'
import re, sys
path = sys.argv[1]
svg = open(path).read()
tag = re.search(r"<svg\b[^>]*>", svg)
scaled = re.sub(
    r'\b(width|height)="([\d.]+)pt"',
    lambda m: f'{m.group(1)}="{3 * float(m.group(2)):.2f}pt"',
    tag.group(0),
)
open(path, "w").write(svg[: tag.start()] + scaled + svg[tag.end() :])
EOF
done

echo "imported into $assets"

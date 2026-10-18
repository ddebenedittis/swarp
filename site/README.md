# swarp project page

Project page for *swarp: A GPU-Resident Differentiable Simulator for High-Throughput Multi-Robot Learning*.

**Live site:** https://ddebenedittis.github.io/swarp/

The page tracks the paper and the [documentation](https://ddebenedittis.github.io/swarp/docs/) tracks `main`.
Numbers on this page are the paper's, so they stay put when `docs/benchmarks.md` moves; anything that is not in the paper belongs in the docs, and the page links there instead.

## Editing

All page content lives in a single file, [`src/paper.mdx`](src/paper.mdx): the frontmatter sets the browser-tab title and the link-preview card, the `<Header>` props set the on-page title, authors, venue and link buttons, and the Markdown body is the page.
Figures and videos go in `src/assets/` and are imported as ES modules so their URLs stay correct under the site's base path.
Only `favicon.svg` and `cover_pic.png` belong in `public/`.

At acceptance: add the `conference` prop and the Paper/arXiv buttons in the `<Header>`, swap the `@misc` BibTeX for the `@article` entry, and pin the tagged release and its Zenodo DOI in the reproduction section.
The placeholders are comments in `paper.mdx`.

## Assets

- `src/assets/scenes/*.png` and `src/assets/ringroad_{scaling,transfer}.svg` are the paper's own figures, copied by `scripts/import_paper_figures.sh /path/to/paper` (needs `pdftocairo` from poppler-utils).
- `src/assets/teaser.webm`, the header background, is rendered by `python site/scripts/make_teaser.py` from the repository root (needs the `viz` extra).
- `src/assets/ringroad_transfer.webm` is the paper's Video S1, copied from `docs/_static/`.

## Development

With Node 22.12 or newer on the host:

```bash
npm ci
npm run dev             # dev server on http://localhost:4321
```

Without Node, everything runs in Docker:

```bash
./docker/build.bash     # build the image (once, and after dependency changes)
./docker/run.bash       # dev server on http://localhost:4321
```

To reproduce the production build, including the `/swarp/` base path that GitHub Pages serves under:

```bash
npx astro build --site "https://ddebenedittis.github.io" --base "/swarp/"
```

## Deployment

Pushing to `main` triggers `../.github/workflows/docs.yml`, which builds this page and the Sphinx docs and publishes them as one GitHub Pages site: this page at the root, the docs under `docs/`.
Each of the docs' old root URLs (`/swarp/<page>.html`) gets a redirect stub to `/swarp/docs/<page>.html`.

## Credits

Built with [Roman Hauksson-Neill's project page template](https://research-template.roman.technology), adapted from [Eliahu Horwitz's template](https://github.com/eliahuhorwitz/Academic-project-page-template), which was adapted from [Keunhong Park's project page for *Nerfies*](https://nerfies.github.io/).
Licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

# Getting Started

This site is built with MkDocs Material and designed for GitHub Pages deployment.

## Local Preview

Install docs dependencies:

```bash
python -m pip install --upgrade pip
pip install mkdocs mkdocs-material pymdown-extensions
```

Start a local server:

```bash
mkdocs serve
```

Open <http://127.0.0.1:8000>.

## Build Static Site

```bash
mkdocs build --strict
```

The generated static site is written to `site/`.

## Deploy with GitHub Pages

This repository includes `.github/workflows/deploy-blog.yml`.

1. Push changes to `main` (or `master`).
2. In repository settings, set Pages source to `GitHub Actions`.
3. The workflow builds with MkDocs and publishes the `site/` directory.

## Authoring Notes

- Keep scientific explanations close to the implementation and link to concrete functions.
- Prefer equations for method definitions and short code snippets for execution flow.
- When behavior changes in code, update the matching blog article in `docs/blog/`.

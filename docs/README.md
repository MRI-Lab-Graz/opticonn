This folder contains the MkDocs-based documentation for the OptiConn project.

Quick local preview

1. (Optional) Activate your project's virtual environment:

   source braingraph_pipeline/bin/activate

2. Install docs dependencies (if needed):

   python -m pip install --upgrade mkdocs mkdocs-material

3. Serve docs locally:

   # from repository root
   ./scripts/docs_serve.sh --port 8000

4. Or build the static site:

   make docs-build

Notes
- The site is configured by `mkdocs.yml` at the repository root.
- `mkdocs-material` is optional but gives a nicer theme; the default `readthedocs` theme works without it.

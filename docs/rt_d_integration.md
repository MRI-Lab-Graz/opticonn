**ReadTheDocs + GitHub integration**

This page documents the minimal steps to connect this repository to ReadTheDocs and to add a GitHub Action that builds the docs on push/PR.

1. Prepare the repo

- Ensure `mkdocs.yml` is present at the repository root and that `docs/` contains your site content.
- We've added `docs/requirements.txt` and `.readthedocs.yml` to this repository so RTD can build the site with MkDocs (Python 3.10).

2. Connect the repo on ReadTheDocs

- Sign in to https://readthedocs.org/ (use your GitHub account).
- Go to "Import a Project" and select this repository (you may need to grant RTD access to your GitHub org/user).
- For build settings, RTD will auto-detect `.readthedocs.yml`. Confirm Python version (3.10) and the install command (it will read `docs/requirements.txt`).

3. Triggering builds from GitHub

- RTD will build on push by default once connected. No extra webhook is required beyond the OAuth integration.
- For faster CI feedback, we've added a GitHub Action (`.github/workflows/docs.yml`) that runs `mkdocs build --clean` on pushes and PRs against `main`. The Action uploads the `site/` artifact so reviewers can download and inspect the built site.

4. Optional: Deploy to GitHub Pages (alternative to RTD)

- If you prefer GitHub Pages, add a step to the workflow that uses `peaceiris/actions-gh-pages` to publish the `site/` directory.

5. Troubleshooting

- If RTD fails to install dependencies, open the project settings on RTD and check the "Advanced Settings" -> "Python configuration" and the build logs.
- If you use an alternative theme/plugin that RTD doesn't allow, pin versions in `docs/requirements.txt` or build in CI to inspect failures locally.

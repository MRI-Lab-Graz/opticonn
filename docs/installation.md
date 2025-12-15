# Installation

## Prerequisites
- macOS or Linux (Windows not supported in this release)
- Python 3.10+
- Git and build tools (Xcode CLT on macOS, build-essential on Linux)
- DSI Studio installed locally (Required). Download: https://github.com/frankyeh/DSI-Studio/releases

## Quick install

1) clone
```console
git clone https://github.com/MRI-Lab-Graz/opticonn.git
cd opticonn
```

2) install with DSI path

```console
# MacOS
bash install.sh --dsi-path /Applications/dsi_studio.app/Contents/MacOS/dsi_studio
```

```console
# Linux
bash install.sh --dsi-path /usr/local/bin/dsi_studio 
```

3) activate
```console
source braingraph_pipeline/bin/activate
```

4) Verify

```console
source braingraph_pipeline/bin/activate
python scripts/validate_setup.py --config configs/braingraph_default_config.json
```

## Notes
- `--dsi-path` is required and must point to the DSI Studio executable.
- If DSI Studio is not found, errors will include the download link above.
- The install uses `uv` to populate the curated virtual environment.

## Docker (reproducible build)

To simulate a fresh environment, build OptiConn in a clean Docker image. The build downloads Python packages during `docker build`. The image does **not** bundle or download DSI Studio.

```console
docker build --target runtime -t opticonn:runtime .
docker run --rm opticonn:runtime --help
```

Apple Silicon note: if you plan to use an x86_64 (amd64) DSI Studio Linux binary inside Docker, build/run with `--platform=linux/amd64`:

```console
docker build --platform=linux/amd64 --target runtime -t opticonn:runtime-amd64 .
docker run --rm --platform=linux/amd64 opticonn:runtime-amd64 --help
```

Docs image (includes MkDocs dependencies):

```console
docker build --target docs -t opticonn:docs .
docker run --rm -p 8000:8000 opticonn:docs
```

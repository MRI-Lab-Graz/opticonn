# Installation

## Prerequisites
- macOS or Linux (Windows not supported in this release)
- Python 3.10+
- Git and build tools (Xcode CLT on macOS, build-essential on Linux)
- DSI Studio installed locally (Required). Download: https://github.com/frankyeh/DSI-Studio/releases

## Quick install
```bash
# clone
git clone https://github.com/MRI-Lab-Graz/opticonn.git
cd opticonn

# install with DSI path
bash install.sh --dsi-path /Applications/dsi_studio.app/Contents/MacOS/dsi_studio   # macOS example
# bash install.sh --dsi-path /usr/local/bin/dsi_studio                             # Linux example

# activate
source braingraph_pipeline/bin/activate
```

## Verify
```bash
source braingraph_pipeline/bin/activate
python scripts/validate_setup.py --config configs/braingraph_default_config.json
```

## Notes
- `--dsi-path` is required and must point to the DSI Studio executable.
- If DSI Studio is not found, errors will include the download link above.
- The install uses `uv` to populate the curated virtual environment.

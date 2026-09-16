# Windows (via WSL2 / Ubuntu)

OptiConn's installer and pipeline scripts assume a POSIX (bash) environment
and are tested on macOS and Linux. There is no native Windows build. The
supported way to run OptiConn on Windows is inside **WSL2 with Ubuntu**,
which gives you a real Ubuntu userspace — the normal Linux install
instructions then apply unchanged.

## 1. Install WSL2 + Ubuntu

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot if prompted, then launch "Ubuntu" from the Start menu and create your
Unix username/password when asked.

If WSL is already installed but you need Ubuntu specifically:

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default Ubuntu-22.04
```

## 2. Install build tools inside Ubuntu

In the Ubuntu terminal:

```bash
sudo apt update
sudo apt install -y build-essential git python3 python3-venv
```

## 3. Get DSI Studio

Download the **Linux** release from
[DSI Studio releases](https://github.com/frankyeh/DSI-Studio/releases) and
extract it inside the WSL filesystem, e.g. `~/dsi-studio/`. Note the path to
the `dsi_studio` executable — you'll pass it to `install.sh`.

Do not point OptiConn at a Windows `.exe` build of DSI Studio; it must be the
Linux binary running inside WSL.

## 4. Clone and install OptiConn

Clone into your **WSL home directory**, not `/mnt/c/...` — the Linux
filesystem is much faster than accessing Windows drives through WSL, and
avoids path/permission quirks.

```bash
git clone https://github.com/MRI-Lab-Graz/opticonn.git ~/opticonn
cd ~/opticonn
bash install.sh --dsi-path ~/dsi-studio/dsi_studio
```

## 5. Activate and verify

```bash
source braingraph_pipeline/bin/activate
python scripts/validate_setup.py --config configs/braingraph_default_config.json
```

## Using data that lives on your Windows drives

Windows drives are mounted under `/mnt/c/`, `/mnt/d/`, etc. inside WSL, so
you can point `--data-dir` there directly:

```bash
python opticonn.py apply --data-dir /mnt/c/Users/<you>/data --output ~/opticonn-results/run1 ...
```

For large datasets, copy the data into the WSL filesystem first
(`cp -r /mnt/c/Users/<you>/data ~/data`) — I/O through `/mnt/c` is
significantly slower than native WSL storage.

## Troubleshooting

- **`wsl` command not recognized**: your Windows version is too old for
  WSL2; update Windows or install WSL manually following
  [Microsoft's WSL install docs](https://learn.microsoft.com/windows/wsl/install).
- **Permission denied on `/mnt/c/...` files**: prefer copying data into the
  WSL filesystem (see above) rather than working on it in place.
- **Garbled log output / crashes on non-ASCII characters**: make sure you're
  running inside the Ubuntu WSL shell, not a native Windows terminal calling
  a Windows Python — WSL's Ubuntu terminal is UTF-8 by default.

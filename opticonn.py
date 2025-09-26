#!/usr/bin/env python3
"""OptiConn CLI entrypoint.

Bridges legacy ``python opticonn.py`` usage to the modern hub located in
``scripts/opticonn_hub.py`` so that downstream documentation and habits keep
working.
"""

from scripts.opticonn_hub import main


if __name__ == "__main__":
    raise SystemExit(main())

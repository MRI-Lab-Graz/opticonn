#!/usr/bin/env python3
"""
OptiConn CLI entrypoint (temporary alias)
For now, forwards to the existing braingraph.main()
"""
from braingraph import main

if __name__ == '__main__':
    raise SystemExit(main())

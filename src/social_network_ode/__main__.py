"""Command-line module entry point (`python -m social_network_ode`)."""

import sys
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

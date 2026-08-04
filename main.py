"""Root entry point: ``python main.py ...`` (same as ``feature-select``)."""

import sys

from feature_selector.cli import main

if __name__ == "__main__":
    sys.exit(main())

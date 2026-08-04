"""Allow ``python -m feature_selector``."""

import sys

from feature_selector.cli import main

if __name__ == "__main__":
    sys.exit(main())

#  Provides a python client for simply retrieving
#  and processing air quality data from the CHMI OpenData portal.
#  Copyright (C) 2025 chickendrop89

#  This library is free software; you can redistribute it and/or modify it
#  under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.

"""
Test runner script for czech_air_quality library.
"""

import sys
import unittest
import argparse
from pathlib import Path
import coverage as cov

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_tests(test_module: str | None = None, verbosity: int = 2, coverage: bool = False):
    """
    Run the test suite.

    :param test_module: Specific test module to run
    :param verbosity: Verbosity level (0=quiet, 1=normal, 2=verbose)
    :param coverage: Whether to measure code coverage
    """
    cov_instance = None

    if coverage:
        try:
            cov_instance = cov.Coverage(
                source=["src"], omit=["*/__pycache__/*", "*/tests/*"]
            )
            cov_instance.start()
        except ImportError:
            print("Error: coverage package not installed")
            print("Install with: pip install coverage")
            return False

    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=verbosity)

    if test_module:
        try:
            suite = loader.loadTestsFromName(test_module)
        except (ImportError, AttributeError, TypeError) as e:
            print(f"Error loading test module '{test_module}': {e}")
            return False
    else:
        test_dir = PROJECT_ROOT / "tests"
        suite = loader.discover(str(test_dir), pattern="test_*.py")

    result = runner.run(suite)

    if cov_instance is not None:
        cov_instance.stop()
        cov_instance.save()

        print("\n" + "=" * 70)
        print("COVERAGE REPORT")
        print("=" * 70)
        cov_instance.report()

    return result.wasSuccessful()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests for czech_air_quality library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                 # Run all tests
  python run_tests.py -v              # Verbose output
  python run_tests.py --coverage      # With coverage report
  python run_tests.py tests.test_airquality   # Specific module
        """,
    )

    parser.add_argument(
        "test_module",
        nargs="?",
        default=None,
        help="Specific test module to run (e.g., tests.test_airquality)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose test output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Measure code coverage"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output (quiet mode)"
    )
    args = parser.parse_args()

    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1

    success = run_tests(
        test_module=args.test_module, verbosity=verbosity, coverage=args.coverage
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

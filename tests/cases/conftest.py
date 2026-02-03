"""pytest configuration for git-polite tests.

Registers custom markers for scenario-based testing and provides fixtures.
"""

import subprocess
import pytest


def pytest_configure(config):
    """Register custom markers for scenario-based testing."""
    config.addinivalue_line(
        "markers",
        "with_changes: Tests that run in the 'with_changes' scenario (001)"
    )
    config.addinivalue_line(
        "markers",
        "conflict: Tests that run in the 'conflict' scenario (002)"
    )
    config.addinivalue_line(
        "markers",
        "multiple_commits: Tests that run in the 'multiple_commits' scenario (003)"
    )
    config.addinivalue_line(
        "markers",
        "newline_eof: Tests that run in the 'newline_eof' scenario (004)"
    )


@pytest.fixture(autouse=True)
def fresh_repo():
    """Provide a fresh copy of the repository for each test.

    The scenario Dockerfile creates the initial state in /repo_template.
    Before each test, we copy /repo_template to /repo to ensure complete isolation.
    We also chdir to /repo so git commands work correctly.
    After each test, /repo is cleaned up.
    """
    import os

    # Remember original directory
    original_cwd = os.getcwd()

    # Copy template to working directory before test
    subprocess.run(
        ["rm", "-rf", "/repo"],
        check=False
    )
    subprocess.run(
        ["cp", "-r", "/repo_template", "/repo"],
        check=True
    )

    # Change to /repo so git commands work
    os.chdir("/repo")

    yield

    # Return to original directory
    os.chdir(original_cwd)

    # Don't delete /repo to avoid pytest session teardown errors
    # Next test will overwrite it anyway

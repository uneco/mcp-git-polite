"""Tests for applying selective changes to the git index.

Tests apply_one_file functionality for staging specific lines.
"""

import sys
sys.path.insert(0, '/app')

import subprocess

import pytest
from git_polite import list_files, apply_one_file


@pytest.mark.with_changes
def test_apply_single_line_to_modified_file():
    """Test applying a single line change to a modified file."""
    # First, get the numbered changes
    list_result = list_files(
        paths=["README.md"],
        page_token=None,
        page_size_files=50,
        page_size_bytes=30*1024,
        unified=3
    )

    readme = list_result["files"][0]
    numbered_lines = [l for l in readme["lines"] if len(l) >= 4 and l[:4].isdigit()]

    # Extract the first line number
    first_line_num = int(numbered_lines[0][:4])

    # Apply only the first change
    apply_result = apply_one_file("README.md", [first_line_num])

    assert len(apply_result["applied"]) == 1
    assert apply_result["applied"][0]["file"] == "README.md"
    assert apply_result["applied"][0]["applied_count"] == 1
    assert len(apply_result["skipped"]) == 0

    # Verify something was staged
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd="/repo",
        capture_output=True,
        text=True
    )
    assert "README.md" in staged.stdout


@pytest.mark.with_changes
def test_apply_to_untracked_file():
    """Test that we can partially stage an untracked file."""
    # Get the changes for the untracked file
    list_result = list_files(
        paths=["new_file.txt"],
        page_token=None,
        page_size_files=50,
        page_size_bytes=30*1024,
        unified=3
    )

    new_file = list_result["files"][0]
    numbered_lines = [l for l in new_file["lines"] if len(l) >= 4 and l[:4].isdigit()]

    # Get the first line number
    first_line_num = int(numbered_lines[0][:4])

    # Apply the first line of the untracked file
    apply_result = apply_one_file("new_file.txt", [first_line_num])

    assert len(apply_result["applied"]) == 1
    assert apply_result["applied"][0]["file"] == "new_file.txt"

    # Verify the file is now in the index
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd="/repo",
        capture_output=True,
        text=True
    )
    assert "new_file.txt" in staged.stdout


@pytest.mark.with_changes
def test_apply_range_of_lines():
    """Test applying a range of line numbers."""
    list_result = list_files(
        paths=["README.md"],
        page_token=None,
        page_size_files=50,
        page_size_bytes=30*1024,
        unified=3
    )

    readme = list_result["files"][0]
    numbered_lines = [l for l in readme["lines"] if len(l) >= 4 and l[:4].isdigit()]

    if len(numbered_lines) >= 2:
        # Apply first two lines
        first = int(numbered_lines[0][:4])
        second = int(numbered_lines[1][:4]) if len(numbered_lines) > 1 else first

        apply_result = apply_one_file("README.md", [first, second])

        assert len(apply_result["applied"]) == 1
        applied_count = apply_result["applied"][0]["applied_count"]
        assert applied_count == 2 or applied_count == 1  # Depends on how many lines exist


@pytest.mark.with_changes
def test_apply_result_structure():
    """Test that apply_one_file returns the expected structure with after_applying."""
    # Get changes
    list_result = list_files(
        paths=["README.md"],
        page_token=None,
        page_size_files=50,
        page_size_bytes=30*1024,
        unified=3
    )

    readme = list_result["files"][0]
    numbered_lines = [l for l in readme["lines"] if len(l) >= 4 and l[:4].isdigit()]

    if numbered_lines:
        first_line_num = int(numbered_lines[0][:4])

        # Apply the change
        apply_result = apply_one_file("README.md", [first_line_num])

        # Check structure
        assert "applied" in apply_result
        assert "skipped" in apply_result
        assert "stats" in apply_result

        if apply_result["applied"]:
            applied = apply_result["applied"][0]
            assert "file" in applied
            assert "applied_count" in applied
            assert "after_applying" in applied

            after_applying = applied["after_applying"]
            assert "diff" in after_applying
            assert "unstaged_lines" in after_applying
            assert isinstance(after_applying["diff"], list)
            assert isinstance(after_applying["unstaged_lines"], int)


@pytest.mark.newline_eof
def test_apply_changes_file_without_trailing_newline():
    """Test applying changes to a file without trailing newline.

    This reproduces the bug where files without trailing newlines
    are not correctly staged by apply_changes.
    """
    # Get the changes
    list_result = list_files(
        paths=["no_newline.txt"],
        page_token=None,
        page_size_files=50,
        page_size_bytes=30*1024,
        unified=3
    )

    assert len(list_result["files"]) == 1
    file_info = list_result["files"][0]
    numbered_lines = [line for line in file_info["lines"] if len(line) >= 4 and line[:4].isdigit()]

    assert len(numbered_lines) > 0, "Should have at least one change"

    # Apply the first change
    first_line_num = int(numbered_lines[0][:4])
    apply_result = apply_one_file("no_newline.txt", [first_line_num])

    # Verify the change was applied
    assert len(apply_result["applied"]) == 1
    assert apply_result["applied"][0]["file"] == "no_newline.txt"

    # Verify it was staged correctly
    staged = subprocess.run(
        ["git", "diff", "--cached", "no_newline.txt"],
        cwd="/repo",
        capture_output=True,
        text=True,
        check=True
    )
    assert len(staged.stdout) > 0, "Should have staged changes"

    # Verify the staged content doesn't have an incorrectly added newline
    staged_content = subprocess.run(
        ["git", "show", ":no_newline.txt"],
        cwd="/repo",
        capture_output=True,
        text=True,
        check=True
    )
    # The file should NOT end with a newline
    assert not staged_content.stdout.endswith("\n"), "Staged file should not have trailing newline"


@pytest.mark.newline_eof
def test_apply_changes_file_with_trailing_newline():
    """Test applying changes to a file with trailing newline.

    This ensures that files with trailing newlines are handled correctly.
    """
    # Get the changes
    list_result = list_files(
        paths=["with_newline.txt"],
        page_token=None,
        page_size_files=50,
        page_size_bytes=30*1024,
        unified=3
    )

    assert len(list_result["files"]) == 1
    file_info = list_result["files"][0]
    numbered_lines = [line for line in file_info["lines"] if len(line) >= 4 and line[:4].isdigit()]

    assert len(numbered_lines) > 0, "Should have at least one change"

    # Apply the first change
    first_line_num = int(numbered_lines[0][:4])
    apply_result = apply_one_file("with_newline.txt", [first_line_num])

    # Verify the change was applied
    assert len(apply_result["applied"]) == 1
    assert apply_result["applied"][0]["file"] == "with_newline.txt"

    # Verify it was staged correctly
    staged = subprocess.run(
        ["git", "diff", "--cached", "with_newline.txt"],
        cwd="/repo",
        capture_output=True,
        text=True,
        check=True
    )
    assert len(staged.stdout) > 0, "Should have staged changes"

    # Verify the staged content has the trailing newline preserved
    staged_content = subprocess.run(
        ["git", "show", ":with_newline.txt"],
        cwd="/repo",
        capture_output=True,
        text=True,
        check=True
    )
    # The file SHOULD end with a newline
    assert staged_content.stdout.endswith("\n"), "Staged file should have trailing newline"


@pytest.mark.newline_eof
def test_apply_changes_removing_trailing_newline():
    """Test applying a change that removes the trailing newline.

    This tests the edge case where the diff removes the trailing newline.
    """
    # Get the changes
    list_result = list_files(
        paths=["remove_newline.txt"],
        page_token=None,
        page_size_files=50,
        page_size_bytes=30*1024,
        unified=3
    )

    assert len(list_result["files"]) == 1
    file_info = list_result["files"][0]
    numbered_lines = [line for line in file_info["lines"] if len(line) >= 4 and line[:4].isdigit()]

    if numbered_lines:
        # Apply all changes (since removing newline might not be separately numbered)
        all_line_nums = [int(line[:4]) for line in numbered_lines]
        apply_result = apply_one_file("remove_newline.txt", all_line_nums)

        # Verify the change was applied
        assert len(apply_result["applied"]) == 1
        assert apply_result["applied"][0]["file"] == "remove_newline.txt"

        # Verify the staged content doesn't have a trailing newline
        staged_content = subprocess.run(
            ["git", "show", ":remove_newline.txt"],
            cwd="/repo",
            capture_output=True,
            text=True,
            check=True
        )
        assert not staged_content.stdout.endswith("\n"), "Staged file should not have trailing newline"


@pytest.mark.newline_eof
def test_apply_changes_adding_trailing_newline():
    """Test applying a change that adds a trailing newline.

    This tests the edge case where the diff adds a trailing newline.
    """
    # Get the changes
    list_result = list_files(
        paths=["add_newline.txt"],
        page_token=None,
        page_size_files=50,
        page_size_bytes=30*1024,
        unified=3
    )

    assert len(list_result["files"]) == 1
    file_info = list_result["files"][0]
    numbered_lines = [line for line in file_info["lines"] if len(line) >= 4 and line[:4].isdigit()]

    if numbered_lines:
        # Apply all changes
        all_line_nums = [int(line[:4]) for line in numbered_lines]
        apply_result = apply_one_file("add_newline.txt", all_line_nums)

        # Verify the change was applied
        assert len(apply_result["applied"]) == 1
        assert apply_result["applied"][0]["file"] == "add_newline.txt"

        # Verify the staged content has a trailing newline
        staged_content = subprocess.run(
            ["git", "show", ":add_newline.txt"],
            cwd="/repo",
            capture_output=True,
            text=True,
            check=True
        )
        assert staged_content.stdout.endswith("\n"), "Staged file should have trailing newline"

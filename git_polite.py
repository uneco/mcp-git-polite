#!/usr/bin/env python3
# git_polite.py
# Git line-level staging via MCP
# Usage:
#   uv run git_polite.py list [--paths <path1> <path2> ...] [--page-token <token>] [--page-size-files N] [--unified N]
#   uv run git_polite.py apply scripts/run-db-seeder.sh 0001,0004,0010-0015
#   uv run git_polite.py mcp  # Run as MCP server

import argparse
import base64
import dataclasses
import json
import os
import re
import stat
import subprocess
import sys
from typing import Any

UNIFIED_LIST_DEFAULT = 20 # Default context width for list
UNIFIED_APPLY = 3 # Context width for apply (fixed)
PAGE_SIZE_FILES_DEFAULT = 50 # Default max files per page (safety limit)
PAGE_SIZE_FILES_MAX = 1000 # Maximum files for batch operations
PAGE_SIZE_BYTES_DEFAULT = 30 * 1024  # default page size in bytes (primary pagination metric)
MAX_DIFF_BYTES = 10 * 1024  # diffs larger than this are truncated (to protect LLM context)

# ---------- Utility ----------

def run(cmd: list[str], cwd: str | None = None, check: bool = True, text: bool = True, input_text: str | None = None) -> str:
    env = os.environ.copy()
    env.setdefault("LC_ALL", "C")
    env.setdefault("LANG", "C")
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=text, input=input_text)
    if check and p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, p.stdout, p.stderr)
    return p.stdout

def git_index_entry(path: str) -> tuple[str | None, str | None]:
    out = run(["git", "ls-files", "-s", "--", path], check=False)
    if not out.strip():
        return None, None
    parts = out.strip().split()
    if len(parts) >= 4:
        return parts[0], parts[1]
    return None, None

def git_read_index_text(path: str) -> tuple[list[str], bool]:
    out = run(["git", "show", f":{path}"], check=False)
    if out is None:
        return [], False
    had_trailing_nl = out.endswith("\n")
    return out.splitlines(keepends=False), had_trailing_nl

def detect_mode_for_path(path: str, fallback_mode: str = "100644") -> str:
    mode, _ = git_index_entry(path)
    if mode:
        return mode
    try:
        st = os.stat(path)
        if st.st_mode & stat.S_IXUSR:
            return "100755"
    except FileNotFoundError:
        pass
    return fallback_mode

def update_index_with_content(path: str, mode: str, content: str) -> None:
    oid = run(["git", "hash-object", "-w", "--stdin"], input_text=content).strip()
    run(["git", "update-index", "--add", "--cacheinfo", f"{mode}", f"{oid}", f"{path}"])

# ---------- diff parsing ----------

@dataclasses.dataclass
class HunkRaw:
    path: str
    header: str        # "@@ -a,b +c,d @@"
    all_lines: list[str]  # All lines with context (' ', '+', '-')
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    old_missing_final_newline: bool = False  # True if old file lacks trailing newline
    new_missing_final_newline: bool = False  # True if new file lacks trailing newline

HUNK_RE = re.compile(r'^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@')

def parse_unified_diff(diff_text: str) -> tuple[dict[str, list[HunkRaw]], dict[str, bool]]:
    """Parse unified diff format into structured hunks per file.

    Args:
        diff_text: Git unified diff output as string

    Returns:
        Tuple of:
        - dict mapping file paths to list of HunkRaw objects
        - dict mapping file paths to binary flag (True if binary file)

    Note:
        Handles standard git diff format including:
        - Multiple files in one diff
        - Binary file detection
        - Omitted line counts in hunk headers (@@ -1 +1 @@ means 1 line)
    """
    files_hunks: dict[str, list[HunkRaw]] = {}
    binaries: dict[str, bool] = {}
    cur_path: str | None = None
    cur_hunk: HunkRaw | None = None

    a_path = b_path = None
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git "):
            a_path = b_path = None
            cur_path = None
            cur_hunk = None
            continue
        if raw.startswith("--- "):
            a_path = raw[4:].strip()
            continue
        if raw.startswith("+++ "):
            b_path = raw[4:].strip()
            if b_path.startswith("b/"):
                cur_path = b_path[2:]
            elif b_path == "/dev/null" and a_path and a_path.startswith("a/"):
                cur_path = a_path[2:]
            else:
                cur_path = b_path
            files_hunks.setdefault(cur_path, [])
            binaries.setdefault(cur_path, False)
            continue
        if raw.startswith("Binary files "):
            m = re.search(r' and b/(.+) differ$', raw)
            if m:
                matched_path = m.group(1)
                cur_path = matched_path
                binaries[matched_path] = True
                files_hunks.setdefault(matched_path, [])
            continue

        m = HUNK_RE.match(raw)
        if m and cur_path:
            old_start = int(m.group(1) or "0")
            old_lines = int(m.group(2) or "1")
            new_start = int(m.group(3) or "0")
            new_lines = int(m.group(4) or "1")
            cur_hunk = HunkRaw(
                path=cur_path,
                header=raw,
                all_lines=[],
                old_start=old_start,
                old_lines=old_lines,
                new_start=new_start,
                new_lines=new_lines,
            )
            files_hunks[cur_path].append(cur_hunk)
            continue

        if cur_hunk is not None and raw:
            if raw.startswith("\\ No newline at end of file"):
                # Track which side is missing the final newline
                # The marker appears after the line that's missing the newline
                if cur_hunk.all_lines:
                    last_line = cur_hunk.all_lines[-1]
                    if last_line[0] == '-':
                        cur_hunk.old_missing_final_newline = True
                    elif last_line[0] == '+':
                        cur_hunk.new_missing_final_newline = True
                continue
            if raw[0] in " +-":
                cur_hunk.all_lines.append(raw)

    return files_hunks, binaries

# ---------- Untracked files support ----------

def get_diff_with_untracked(paths: list[str], unified: int) -> tuple[str, set, set]:
    """Get git diff including untracked files.

    Returns:
        Tuple of (diff_text, set of untracked file paths, set of deleted file paths)
    """
    # Get regular diff for tracked files
    diff_cmd = ["git", "diff", "--patch", f"--unified={unified}", "--no-color", "--no-ext-diff", "--find-renames=50%"]
    if paths:
        diff_cmd += ["--"] + paths
    diff_text = run(diff_cmd)

    # Get untracked files and generate diffs for them
    untracked_cmd = ["git", "ls-files", "--others", "--exclude-standard"]
    if paths:
        untracked_cmd += ["--"] + paths
    untracked_files = run(untracked_cmd, check=False).strip().split("\n")
    untracked_files = list(filter(None, untracked_files))  # Filter empty lines
    untracked_set = set(untracked_files)

    # Get deleted files
    deleted_cmd = ["git", "ls-files", "--deleted"]
    if paths:
        deleted_cmd += ["--"] + paths
    deleted_files = run(deleted_cmd, check=False).strip().split("\n")
    deleted_files = list(filter(None, deleted_files))
    deleted_set = set(deleted_files)

    # Generate diff for each untracked file (comparing /dev/null to file)
    for untracked_file in untracked_files:
        try:
            # git diff --no-index shows new files as additions from /dev/null
            untracked_diff = run([
                "git", "diff", "--no-index",
                f"--unified={unified}",
                "--no-color",
                "/dev/null",
                untracked_file
            ], check=False)
            # Append untracked file diff to main diff
            if untracked_diff.strip():
                diff_text += "\n" + untracked_diff
        except Exception:
            # Skip files that can't be diffed (binary, etc.)
            pass

    return diff_text, untracked_set, deleted_set

# ---------- Display (flattened) ----------

def calculate_line_stats(hunks: list[HunkRaw]) -> dict[str, int]:
    """Calculate line statistics from hunks without generating full diff.

    Args:
        hunks: List of diff hunks for a file

    Returns:
        Dictionary with keys:
        - additions: Number of lines added
        - deletions: Number of lines deleted
        - changes: Total number of changed lines (additions + deletions)
    """
    additions = 0
    deletions = 0
    for h in hunks:
        for ln in h.all_lines:
            if ln and len(ln) > 0:
                sign = ln[0]
                if sign == '+':
                    additions += 1
                elif sign == '-':
                    deletions += 1
    return {
        "additions": additions,
        "deletions": deletions,
        "changes": additions + deletions
    }

def calculate_diff_size(hunks: list[HunkRaw]) -> int:
    """Calculate the total byte size of diff content in hunks.

    Args:
        hunks: List of diff hunks for a file

    Returns:
        Total byte size of all diff lines (used to detect large diffs)
    """
    total_bytes = 0
    for h in hunks:
        for ln in h.all_lines:
            total_bytes += len(ln.encode('utf-8'))
    return total_bytes

def flat_file_lines_with_numbers(hunks: list[HunkRaw], include_trailing_newline: bool = True) -> list[str]:
    out: list[str] = []
    n = 1  # Sequential number of changed lines (within file)
    first = True
    last_hunk = None
    for h in sorted(hunks, key=lambda x: (x.old_start, x.new_start)):
        if not first:
            out.append("        ...")
        first = False
        for ln in h.all_lines:
            sign = ln[0]
            text = ln[1:]
            if sign in "+-":
                out.append(f"{n:04d}: {sign} {text}")
                n += 1
            elif sign == " ":
                out.append("        " + text)
        last_hunk = h

    # If the new file has a trailing newline and last line was an addition,
    # show an empty line that represents the trailing newline
    if include_trailing_newline and last_hunk and not last_hunk.new_missing_final_newline:
        # Check if the last change was an addition (file has content)
        has_additions = any(ln and ln[0] == '+' for ln in last_hunk.all_lines)
        if has_additions:
            out.append(f"{n:04d}: + ")

    return out

def current_file_lines(path: str, unified: int = UNIFIED_APPLY) -> dict[str, Any]:
    """Get the current diff of the target file in 'lines' format for apply response."""
    diff_text, _, _ = get_diff_with_untracked([path], unified)
    files_hunks, binaries = parse_unified_diff(diff_text)
    binflag = binaries.get(path, False)
    hunks = files_hunks.get(path, [])
    return {
        "path": path,
        "binary": binflag,
        "lines": [] if binflag else flat_file_lines_with_numbers(hunks, include_trailing_newline=True)
    }

# ---------- list (flat output per file) ----------

def list_files(paths: list[str], page_token: str | None, page_size_files: int, page_size_bytes: int = PAGE_SIZE_BYTES_DEFAULT, unified: int = UNIFIED_LIST_DEFAULT) -> dict:
    """List changed files with line-level numbering for selective staging.

    Args:
        paths: List of file paths to filter (empty list = all files)
        page_token: Opaque pagination token from previous call (None = start)
        page_size_files: Maximum number of files per page (safety limit)
        page_size_bytes: Maximum cumulative diff size per page in bytes (primary limit)
        unified: Number of context lines around changes

    Returns:
        Dictionary with keys:
        - page_token_next: Token for next page (None if last page)
        - files: List of file dicts with path, binary, status, lines
          - If truncated=True, the diff was too large and lines will be empty
        - stats: Summary with files, lines, truncated_files, and page_size_bytes

    Note:
        - File status can be: "added" (untracked), "deleted", or "modified"
        - Large diffs (>MAX_DIFF_BYTES) are automatically truncated to protect LLM context
        - Pagination stops when cumulative size exceeds page_size_bytes OR file count exceeds page_size_files
    """
    diff_text, untracked_set, deleted_set = get_diff_with_untracked(paths, unified)
    files_hunks, binaries = parse_unified_diff(diff_text)
    all_paths = sorted(files_hunks.keys())  # Sort for consistent pagination order

    start_idx = 0
    if page_token:
        try:
            # Add padding if needed (base64 requires length to be multiple of 4)
            padding = len(page_token) % 4
            if padding:
                page_token += '=' * (4 - padding)
            st = json.loads(base64.urlsafe_b64decode(page_token).decode("utf-8"))
            start_idx = int(st.get("file_index", 0))
        except (ValueError, json.JSONDecodeError, KeyError, TypeError):
            # Invalid page token, start from beginning
            start_idx = 0

    out_files: list[dict[str, Any]] = []
    truncated_count = 0
    cumulative_bytes = 0
    i = start_idx

    # Iterate through files until we exceed byte limit or file limit
    while i < len(all_paths):
        # Stop if we've reached file count limit
        if len(out_files) >= page_size_files:
            break

        # Stop if we've exceeded byte limit (but always include at least one file)
        if cumulative_bytes > 0 and cumulative_bytes >= page_size_bytes:
            break
        p = all_paths[i]
        hunks = files_hunks[p]
        binflag = binaries.get(p, False)

        # Determine file status
        if p in untracked_set:
            status = "added"
        elif p in deleted_set:
            status = "deleted"
        else:
            status = "modified"

        if binflag:
            out_files.append({"path": p, "binary": True, "status": status, "lines": []})
            # Binary files don't contribute to cumulative size
            i += 1
            continue

        # Check diff size to avoid overwhelming LLM context
        diff_size = calculate_diff_size(hunks)
        if diff_size > MAX_DIFF_BYTES:
            size_kb = diff_size / 1024
            out_files.append({
                "path": p,
                "binary": False,
                "status": status,
                "truncated": True,
                "reason": f"diff too large ({size_kb:.1f} KB, max {MAX_DIFF_BYTES // 1024} KB)",
                "lines": []
            })
            truncated_count += 1
            # Truncated files don't contribute to cumulative size
            i += 1
            continue

        # Include this file and add its size to cumulative total
        lines = flat_file_lines_with_numbers(hunks)
        out_files.append({"path": p, "binary": False, "status": status, "lines": lines})

        # Calculate actual output size (lines with formatting)
        lines_bytes = sum(len(line.encode('utf-8')) for line in lines)
        cumulative_bytes += lines_bytes
        i += 1

    # Create next page token if there are more files
    page_token_next = None
    if i < len(all_paths):
        next_state: dict[str, Any] = {"file_index": i}
        page_token_next = base64.urlsafe_b64encode(json.dumps(next_state).encode("utf-8")).decode("ascii").rstrip("=")

    return {
        "page_token_next": page_token_next,
        "files": out_files,
        "stats": {
            "files": len(out_files),
            "lines": sum(len(f.get("lines", [])) for f in out_files if not f.get("binary", False)),
            "truncated_files": truncated_count,
            "page_bytes": cumulative_bytes
        }
    }

# ---------- apply (partial application for 1 file, by number/range) ----------

def count_hunk_changes(hunks: list[HunkRaw]) -> int:
    """Count the total number of changed lines (additions and deletions) in hunks.

    Args:
        hunks: List of diff hunks

    Returns:
        Total count of lines that have line numbers (excluding context lines)
    """
    count = 0
    for hunk in hunks:
        for ln in hunk.all_lines:
            if ln and ln[0] in '+-':
                count += 1
    return count

def determine_new_trailing_newline(hunks: list[HunkRaw], want_numbers: set[int], default: bool) -> bool:
    """Determine if the new file should have a trailing newline.

    Args:
        hunks: List of diff hunks
        want_numbers: Set of line numbers that were selected for application
        default: Default value if no explicit newline information in hunks

    Returns:
        True if new file should have trailing newline, False otherwise
    """
    # Count actual changed lines in hunks
    actual_change_count = count_hunk_changes(hunks)

    # Check if user selected a line number beyond actual changes
    # This indicates they selected the trailing newline line
    max_selected = max(want_numbers) if want_numbers else 0
    trailing_newline_selected = max_selected > actual_change_count

    # Check if any hunk explicitly indicates the new file's trailing newline status
    for hunk in hunks:
        # If new version explicitly lacks trailing newline
        if hunk.new_missing_final_newline:
            # User can override by selecting the trailing newline line
            return trailing_newline_selected
        # If old version lacked trailing newline but new doesn't have the marker,
        # it means trailing newline was added
        if hunk.old_missing_final_newline and not hunk.new_missing_final_newline:
            # If trailing newline line wasn't selected, don't add it
            if not trailing_newline_selected:
                return False
            return True

    # No explicit information about trailing newline changes
    # If user selected trailing newline line, add it
    if trailing_newline_selected:
        return True

    # Otherwise use default (from old file)
    return default

def apply_one_file(path: str, want_numbers: list[int]) -> dict:
    """Apply selected line changes to a single file and stage to git index.

    This function reads the current diff for a file, applies only the selected
    changes (by line number), and updates the git index with the result.

    Args:
        path: File path to apply changes to
        want_numbers: List of change line numbers to apply (1-indexed)

    Returns:
        Dictionary with keys:
        - applied: List of successfully applied changes with file info
          Each entry contains:
          - file: File path
          - applied_count: Number of changes applied
          - after_applying: Object with diff (numbered lines) and unstaged_lines count
        - skipped: List of skipped changes with reasons (binary, drift)
        - stats: Summary statistics (files, changes_applied, changes_skipped)

    Note:
        The function handles:
        - Binary files (skips with reason "binary")
        - Drift detection (skips with reason "drift" if file changed)
        - Untracked files (creates new index entry)
    """
    want_set = set(want_numbers)

    diff_text, _, _ = get_diff_with_untracked([path], UNIFIED_APPLY)
    files_hunks, binaries = parse_unified_diff(diff_text)

    if binaries.get(path, False):
        return {
            "applied": [],
            "skipped": [{"file": path, "number": n, "reason": "binary"} for n in sorted(want_set)],
            "stats": {"files": 0, "changes_applied": 0, "changes_skipped": len(want_set)}
        }

    hunks = files_hunks.get(path, [])
    if not hunks:
        return {
            "applied": [],
            "skipped": [{"file": path, "number": n, "reason": "drift"} for n in sorted(want_set)],
            "stats": {"files": 0, "changes_applied": 0, "changes_skipped": len(want_set)}
        }

    old_lines: list[str]
    had_trailing_nl: bool
    old_lines, had_trailing_nl = git_read_index_text(path)

    try:
        new_lines = apply_selected_changes_to_old(old_lines, hunks, want_set)
    except ValueError:
        return {
            "applied": [],
            "skipped": [{"file": path, "number": n, "reason": "drift"} for n in sorted(want_set)],
            "stats": {"files": 0, "changes_applied": 0, "changes_skipped": len(want_set)}
        }

    mode = detect_mode_for_path(path)
    new_text = "\n".join(new_lines)
    # Determine if new file should have trailing newline based on diff info and selected lines
    new_has_trailing_nl = determine_new_trailing_newline(hunks, want_set, had_trailing_nl)
    if new_has_trailing_nl:
        new_text += "\n"
    update_index_with_content(path, mode, new_text)

    file_info = current_file_lines(path)
    # Count remaining unstaged changes (lines that start with 4-digit number)
    unstaged_count = sum(1 for line in file_info["lines"] if line and len(line) >= 4 and line[:4].isdigit())
    return {
        "applied": [{
            "file": path,
            "applied_count": len(want_set),
            "after_applying": {
                "diff": file_info["lines"],
                "unstaged_lines": unstaged_count
            }
        }],
        "skipped": [],
        "stats": {"files": 1, "changes_applied": len(want_set), "changes_skipped": 0}
    }

def apply_selected_changes_to_old(old_lines: list[str], hunks: list[HunkRaw], want_numbers: set) -> list[str]:
    """Apply selected changes from diff hunks to old file content.

    This function takes the original file content and applies only the changes
    specified by their sequential numbers. This enables partial staging of changes.

    Args:
        old_lines: Original file content as list of lines (without newlines)
        hunks: List of diff hunks to process, sorted by position
        want_numbers: Set of change numbers (1-indexed) to apply

    Returns:
        New file content with selected changes applied

    Raises:
        ValueError: If hunks don't match the old file content (drift detected)

    Example:
        old_lines = ["line 1", "line 2", "line 3"]
        # Hunk that adds "new line" after line 2
        hunks = [HunkRaw(...)]
        want_numbers = {1}  # Apply only the first change
        result = apply_selected_changes_to_old(old_lines, hunks, want_numbers)
        # result = ["line 1", "line 2", "new line", "line 3"]
    """
    new: list[str] = []
    old_pos = 1  # 1-origin
    num_counter = 1  # Sequential number of changed lines

    for h in sorted(hunks, key=lambda x: (x.old_start, x.new_start)):
        pre_start = h.old_start if h.old_start > 0 else 1
        if pre_start - 1 > len(old_lines) + 1:
            raise ValueError(f"Hunk old_start={pre_start} is out of bounds (file has {len(old_lines)} lines)")
        while old_pos < pre_start:
            if old_pos - 1 >= len(old_lines):
                break
            new.append(old_lines[old_pos - 1])
            old_pos += 1

        for ln in h.all_lines:
            if not ln:
                continue  # Skip empty lines
            sign = ln[0]
            text = ln[1:]

            if sign == " ":
                if old_pos - 1 >= len(old_lines):
                    raise ValueError(f"Context line at position {old_pos} is beyond file end (file has {len(old_lines)} lines)")
                if old_lines[old_pos - 1] != text:
                    raise ValueError(
                        f"Context mismatch at line {old_pos}:\n"
                        f"  Expected: {repr(text)}\n"
                        f"  Got:      {repr(old_lines[old_pos - 1])}"
                    )
                new.append(old_lines[old_pos - 1])
                old_pos += 1

            elif sign == "-":
                if old_pos - 1 >= len(old_lines):
                    raise ValueError(f"Deletion at position {old_pos} is beyond file end (file has {len(old_lines)} lines)")
                if old_lines[old_pos - 1] != text:
                    raise ValueError(
                        f"Deletion mismatch at line {old_pos}:\n"
                        f"  Expected to delete: {repr(text)}\n"
                        f"  Found:              {repr(old_lines[old_pos - 1])}"
                    )
                if num_counter in want_numbers:
                    pass  # consume only (apply deletion)
                else:
                    new.append(old_lines[old_pos - 1])  # keep if not selected
                old_pos += 1
                num_counter += 1

            elif sign == "+":
                if num_counter in want_numbers:
                    new.append(text)  # apply addition
                # don't insert if not selected
                num_counter += 1

            else:
                raise ValueError(f"Unexpected diff line marker '{sign}' at line {old_pos} (expected ' ', '+', or '-')")

    while old_pos - 1 < len(old_lines):
        new.append(old_lines[old_pos - 1])
        old_pos += 1

    return new

# ---------- ANSI Colors ----------

ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_CYAN = "\033[36m"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"

def format_pretty(result: dict) -> str:
    """Format list_files result as colored plain text.

    Args:
        result: Dictionary returned by list_files()

    Returns:
        Colored plain text string with:
        - Green for added lines (+)
        - Red for deleted lines (-)
        - Cyan for file headers
    """
    lines: list[str] = []

    for file_info in result.get("files", []):
        path = file_info.get("path", "")
        status = file_info.get("status", "modified")
        binary = file_info.get("binary", False)
        truncated = file_info.get("truncated", False)

        # File header
        status_label = f" ({status})" if status != "modified" else ""
        lines.append(f"{ANSI_CYAN}{ANSI_BOLD}{path}{status_label}{ANSI_RESET}")

        if binary:
            lines.append("  (binary file)")
            lines.append("")
            continue

        if truncated:
            reason = file_info.get("reason", "diff too large")
            lines.append(f"  (truncated: {reason})")
            lines.append("")
            continue

        # Diff lines
        for line in file_info.get("lines", []):
            if line.startswith("        ..."):
                lines.append(line)
            elif len(line) >= 7 and line[4] == ':' and line[6] in '+-':
                sign = line[6]
                if sign == '+':
                    lines.append(f"{ANSI_GREEN}{line}{ANSI_RESET}")
                elif sign == '-':
                    lines.append(f"{ANSI_RED}{line}{ANSI_RESET}")
                else:
                    lines.append(line)
            else:
                lines.append(line)

        lines.append("")

    # Stats
    stats = result.get("stats", {})
    files_count = stats.get("files", 0)
    lines_count = stats.get("lines", 0)
    truncated_count = stats.get("truncated_files", 0)
    lines.append(f"--- {files_count} file(s), {lines_count} line(s)")
    if truncated_count > 0:
        lines.append(f"    ({truncated_count} file(s) truncated)")

    # Pagination
    if result.get("page_token_next"):
        lines.append("    (more pages available)")

    return "\n".join(lines)

def format_apply_pretty(result: dict) -> str:
    """Format apply_one_file result as colored plain text.

    Args:
        result: Dictionary returned by apply_one_file()

    Returns:
        Colored plain text string showing applied changes and remaining diff
    """
    lines: list[str] = []

    # Error handling
    if "error" in result:
        lines.append(f"{ANSI_RED}Error: {result['error']}{ANSI_RESET}")
        return "\n".join(lines)

    # Applied files
    for applied in result.get("applied", []):
        file_path = applied.get("file", "")
        applied_count = applied.get("applied_count", 0)
        after_applying = applied.get("after_applying", {})
        unstaged = after_applying.get("unstaged_lines", 0)

        lines.append(f"{ANSI_GREEN}Applied {applied_count} change(s) to {file_path}{ANSI_RESET}")
        if unstaged > 0:
            lines.append(f"  {unstaged} unstaged change(s) remaining")

        # Show remaining diff
        file_lines = after_applying.get("diff", [])
        if file_lines:
            lines.append("")
            lines.append(f"{ANSI_CYAN}{ANSI_BOLD}Remaining changes in {file_path}:{ANSI_RESET}")
            for line in file_lines:
                if line.startswith("        ..."):
                    lines.append(line)
                elif len(line) >= 7 and line[4] == ':' and line[6] in '+-':
                    sign = line[6]
                    if sign == '+':
                        lines.append(f"{ANSI_GREEN}{line}{ANSI_RESET}")
                    elif sign == '-':
                        lines.append(f"{ANSI_RED}{line}{ANSI_RESET}")
                    else:
                        lines.append(line)
                else:
                    lines.append(line)

    # Skipped files
    for skipped in result.get("skipped", []):
        file_path = skipped.get("file", "")
        number = skipped.get("number", "")
        reason = skipped.get("reason", "unknown")
        lines.append(f"{ANSI_RED}Skipped {file_path} #{number}: {reason}{ANSI_RESET}")

    # Stats summary
    stats = result.get("stats", {})
    applied_count = stats.get("changes_applied", 0)
    skipped_count = stats.get("changes_skipped", 0)
    lines.append("")
    lines.append(f"--- {applied_count} applied, {skipped_count} skipped")

    return "\n".join(lines)

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(prog="git-polite", description="Git line-level staging")
    sub = p.add_subparsers(dest="cmd", required=True)

    list_parser = sub.add_parser("list", help="List file diffs as flat 'lines' with context")
    list_parser.add_argument("--paths", nargs="*", default=[], help="Paths to include (default: all)")
    list_parser.add_argument("--page-token", default=None, help="Opaque paging token (files only)")
    list_parser.add_argument("--page-size-files", type=int, default=PAGE_SIZE_FILES_DEFAULT, help="Max files per page")
    list_parser.add_argument("--page-size-bytes", type=int, default=PAGE_SIZE_BYTES_DEFAULT, help="Max bytes per page")
    list_parser.add_argument("--unified", type=int, default=UNIFIED_LIST_DEFAULT, help="Context lines around hunks")
    list_parser.add_argument("--format", choices=["json", "pretty"], default="json", help="Output format (default: json)")

    a = sub.add_parser("apply", help="Apply selected change numbers for a single file")
    a.add_argument("path", help="Target file path")
    a.add_argument("numbers", help="NNNN,MMMM,PPPP-QQQQ format change numbers to apply")
    a.add_argument("--format", choices=["json", "pretty"], default="json", help="Output format (default: json)")

    sub.add_parser("mcp", help="Run as MCP server (stdio)")

    return p.parse_args()

def parse_number_tokens(token_str: str) -> list[int]:
    nums: list[int] = []
    for tok in token_str.split(","):
        t = tok.strip()
        if not t:
            continue
        if re.fullmatch(r"\d{4}", t):
            nums.append(int(t))
            continue
        m = re.fullmatch(r"(\d{4})-(\d{4})", t)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                raise ValueError(f"invalid range: {t}")
            nums.extend(range(a, b + 1))
            continue
        raise ValueError(f"invalid token: {t}")
    return nums

# ---------- unstack implementation ----------

def do_unstack(branches: dict[str, list[str]], parent: str = "origin/main") -> dict:
    """Unstack linear commits into parallel branches.

    Implementation of the unstack functionality that can be tested independently.

    Args:
        branches: Dictionary mapping branch names to lists of commit references
        parent: Base commit to branch from (default: "origin/main")

    Returns:
        Dictionary with created_branches, errors, and stats
    """
    created_branches = []
    errors = []

    # Resolve parent commit SHA
    try:
        parent_sha = run(["git", "rev-parse", parent]).strip()
    except subprocess.CalledProcessError as e:
        return {
            "created_branches": [],
            "errors": [{
                "branch": None,
                "commit": parent,
                "error": f"Failed to resolve parent '{parent}': {e.stderr if e.stderr else str(e)}"
            }],
            "stats": {
                "total_branches": len(branches),
                "successful_branches": 0,
                "failed_branches": len(branches)
            }
        }

    for branch_name, commits in branches.items():
        try:
            # Check if branch already exists
            branch_check = run(["git", "rev-parse", "--verify", f"refs/heads/{branch_name}"], check=False)
            if branch_check.strip():
                errors.append({
                    "branch": branch_name,
                    "commit": None,
                    "error": f"Branch '{branch_name}' already exists"
                })
                continue

            # Start from parent commit
            current_parent = parent_sha
            commits_applied = []

            for commit_ref in commits:
                try:
                    # Resolve commit ref to SHA
                    commit_sha = run(["git", "rev-parse", commit_ref]).strip()

                    # Get the parent of the commit we're cherry-picking
                    original_parent = run(["git", "rev-parse", f"{commit_sha}^"]).strip()

                    # Try to use modern merge-tree with --write-tree (Git 2.38+)
                    # Syntax: git merge-tree --write-tree --merge-base=<base> <branch1> <branch2>
                    merge_tree_result = run(
                        ["git", "merge-tree", "--write-tree", f"--merge-base={original_parent}", current_parent, commit_sha],
                        check=False
                    )

                    # If merge-tree --write-tree is not available (old Git), fall back
                    # Note: errors go to stderr, so we check the result
                    if not merge_tree_result.strip() or "unknown option" in merge_tree_result.lower():
                        # Fallback: use old merge-tree
                        merge_result = run(
                            ["git", "merge-tree", original_parent, current_parent, commit_sha],
                            check=False
                        )
                        if "<<<<<" in merge_result or "=====" in merge_result or ">>>>>" in merge_result:
                            errors.append({
                                "branch": branch_name,
                                "commit": commit_ref,
                                "error": "Merge conflict detected (Git version does not support --write-tree)"
                            })
                            break
                        tree_sha = run(["git", "rev-parse", f"{commit_sha}^{{tree}}"]).strip()
                    else:
                        lines = merge_tree_result.strip().split("\n")
                        if not lines:
                            errors.append({
                                "branch": branch_name,
                                "commit": commit_ref,
                                "error": "Empty merge-tree result"
                            })
                            break

                        # First line should be tree SHA
                        potential_tree = lines[0].strip()

                        # Check if there are conflict markers in the output
                        # merge-tree outputs "CONFLICT" messages after the tree SHA
                        has_conflict = any("CONFLICT" in line for line in lines[1:])

                        if has_conflict:
                            errors.append({
                                "branch": branch_name,
                                "commit": commit_ref,
                                "error": "Merge conflict detected during cherry-pick"
                            })
                            break
                        elif len(potential_tree) == 40 and all(c in "0123456789abcdef" for c in potential_tree):
                            tree_sha = potential_tree
                        else:
                            errors.append({
                                "branch": branch_name,
                                "commit": commit_ref,
                                "error": f"Unexpected merge-tree output: {potential_tree}"
                            })
                            break

                    # Get the original commit message and author
                    commit_message = run(["git", "log", "--format=%B", "-n", "1", commit_sha]).strip()
                    author_name = run(["git", "log", "--format=%an", "-n", "1", commit_sha]).strip()
                    author_email = run(["git", "log", "--format=%ae", "-n", "1", commit_sha]).strip()
                    author_date = run(["git", "log", "--format=%aI", "-n", "1", commit_sha]).strip()

                    # Create new commit with the tree
                    env = os.environ.copy()
                    env["GIT_AUTHOR_NAME"] = author_name
                    env["GIT_AUTHOR_EMAIL"] = author_email
                    env["GIT_AUTHOR_DATE"] = author_date
                    env.setdefault("LC_ALL", "C")
                    env.setdefault("LANG", "C")

                    p = subprocess.run(
                        ["git", "commit-tree", tree_sha, "-p", current_parent, "-m", commit_message],
                        capture_output=True,
                        text=True,
                        env=env
                    )
                    if p.returncode != 0:
                        raise subprocess.CalledProcessError(p.returncode, p.args, p.stdout, p.stderr)
                    new_commit_sha = p.stdout.strip()

                    commits_applied.append({
                        "ref": commit_ref,
                        "original_sha": commit_sha,
                        "new_sha": new_commit_sha
                    })

                    current_parent = new_commit_sha

                except subprocess.CalledProcessError as e:
                    errors.append({
                        "branch": branch_name,
                        "commit": commit_ref,
                        "error": f"Failed to apply commit: {e.stderr if e.stderr else str(e)}"
                    })
                    break
            else:
                # All commits applied successfully, create the branch
                try:
                    run(["git", "update-ref", f"refs/heads/{branch_name}", current_parent])
                    created_branches.append({
                        "name": branch_name,
                        "commits_applied": commits_applied,
                        "head_sha": current_parent
                    })
                except subprocess.CalledProcessError as e:
                    errors.append({
                        "branch": branch_name,
                        "commit": None,
                        "error": f"Failed to create branch ref: {e.stderr if e.stderr else str(e)}"
                    })

        except subprocess.CalledProcessError as e:
            errors.append({
                "branch": branch_name,
                "commit": None,
                "error": f"Failed to process branch: {e.stderr if e.stderr else str(e)}"
            })

    return {
        "created_branches": created_branches,
        "errors": errors,
        "stats": {
            "total_branches": len(branches),
            "successful_branches": len(created_branches),
            "failed_branches": len(errors)
        }
    }

# ---------- MCP Server ----------

def create_mcp_server():
    """Create and configure MCP server with FastMCP."""
    try:
        from fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ImportError:
        print("Error: fastmcp package not found. Install with: pip install fastmcp", file=sys.stderr)
        sys.exit(1)

    mcp = FastMCP("git-polite")

    @mcp.tool(annotations=ToolAnnotations(
        readOnlyHint=True,
        openWorldHint=True
    ))
    def list_changes(
        paths: list[str] = [],
        page_token: str | None = None,
        page_size_files: int = PAGE_SIZE_FILES_DEFAULT,
        page_size_bytes: int = PAGE_SIZE_BYTES_DEFAULT,
        unified: int = UNIFIED_LIST_DEFAULT
    ) -> str:
        """View unstaged git changes with line-level selection numbers for partial staging.

        PREFER THIS OVER `git diff` when you need to selectively stage changes. Unlike `git diff`,
        this tool includes untracked files (newly created files) in the output. This tool numbers
        each changed line (0001, 0002, etc.) so you can stage specific lines or ranges instead of
        entire files. Essential for creating multiple logical commits from intermixed changes.

        Key features:
        - Includes untracked files (status: "added") as well as modified files (status: "modified")
        - Numbers every changed line for precise selection
        - Supports byte-based pagination to protect LLM context
        - Auto-truncates large diffs (>10KB) with clear indication

        Handling truncated files:
        When a file shows `truncated: true` with empty `lines: []`, use the `diff` tool to view
        its complete content. The `diff` tool returns the same numbered line format needed for
        partial staging with `apply_changes`, whereas `git diff` output lacks line numbers and
        cannot be used for selective staging. For example, if a large refactored file is truncated,
        call `diff(path="src/large_module.py")` to see the full numbered diff and selectively stage
        related changes.

        Use cases:
        - Breaking up large changes into multiple focused commits
        - Staging only specific changes while keeping others unstaged
        - Creating atomic commits from work-in-progress code
        - Separating refactoring from feature changes
        - Selectively staging parts of newly created files

        After viewing changes, use apply_changes with the line numbers to stage selected changes.

        Args:
            paths: Optional list of file paths to filter (default: all files)
            page_token: Opaque pagination token from previous response
            page_size_files: Max files per page - safety limit (default: PAGE_SIZE_FILES_DEFAULT)
            page_size_bytes: Max bytes per page - primary limit (default: PAGE_SIZE_BYTES_DEFAULT)
            unified: Context lines around changes (default: UNIFIED_LIST_DEFAULT)

        Returns:
            JSON string with format: {page_token_next, files: [{path, binary, lines}], stats}
        """
        result = list_files(paths, page_token, page_size_files, page_size_bytes, unified)
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool(annotations=ToolAnnotations(
        readOnlyHint=True,
        openWorldHint=True
    ))
    def diff(path: str, unified: int = UNIFIED_LIST_DEFAULT) -> str:
        """View complete diff for a single file without truncation.

        This tool is designed for viewing the full diff of a single file, regardless of size.
        Unlike list_changes, this tool will NEVER truncate the output, making it suitable
        for reviewing large files like lock files or generated code.

        Use this when you need to:
        - View the complete diff of a large file (e.g., uv.lock, package-lock.json)
        - Review all changes in a specific file before staging
        - Analyze files that would be truncated by list_changes

        Args:
            path: File path to view diff for (required)
            unified: Context lines around changes (default: UNIFIED_LIST_DEFAULT)

        Returns:
            JSON string with format: {path, binary, status, lines, size_bytes}
        """
        # Get diff for single file only
        diff_text, untracked_set, deleted_set = get_diff_with_untracked([path], unified)
        files_hunks, binaries = parse_unified_diff(diff_text)

        # Check if file has changes
        if path not in files_hunks:
            return json.dumps({
                "path": path,
                "error": "No changes found for this file",
                "size_bytes": 0
            }, ensure_ascii=False, indent=2)

        hunks = files_hunks[path]
        binflag = binaries.get(path, False)

        # Determine file status
        if path in untracked_set:
            status = "added"
        elif path in deleted_set:
            status = "deleted"
        else:
            status = "modified"

        if binflag:
            result = {
                "path": path,
                "binary": True,
                "status": status,
                "lines": [],
                "size_bytes": 0
            }
        else:
            # Generate lines WITHOUT truncation
            lines = flat_file_lines_with_numbers(hunks)
            lines_bytes = sum(len(line.encode('utf-8')) for line in lines)
            result = {
                "path": path,
                "binary": False,
                "status": status,
                "lines": lines,
                "size_bytes": lines_bytes
            }

        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool(annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        openWorldHint=True
    ))
    def apply_changes(path: str, numbers: str) -> str:
        """Stage selected lines to git index for partial commits (alternative to `git add -p`).

        After using list_changes to view numbered changes, use this tool to selectively stage
        specific lines or ranges to the git index. This enables creating multiple logical commits
        from a single file with intermixed changes.

        Unlike `git add`, this tool can stage parts of untracked files (newly created files).
        You can commit only the first 10 lines of a new file while keeping the rest unstaged.

        Number format examples:
        - Single lines: "0001,0002,0005"
        - Ranges: "0001-0010"
        - Combined: "0001-0005,0020-0025"

        The tool updates the git index directly and reports remaining unstaged changes, allowing
        iterative staging for multiple commits from the same file.

        Args:
            path: File path to apply changes to
            numbers: Change numbers in format: NNNN,MMMM,PPPP-QQQQ

        Returns:
            JSON string with format: {applied: [{file, applied_count, after_applying: {diff, unstaged_lines}}], skipped, stats}
        """
        try:
            nums = parse_number_tokens(numbers)
        except ValueError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

        result = apply_one_file(path, nums)
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool(annotations=ToolAnnotations(
        readOnlyHint=True,
        openWorldHint=True
    ))
    def auto_commit() -> str:
        """Start a guided session to organize and commit all unstaged changes with appropriate granularity.

        This tool helps you organize your changes and create multiple focused commits by:
        1. Showing recent commit messages as style reference
        2. Displaying summary statistics of all unstaged changes (file counts, line counts)
        3. Providing step-by-step instructions for the commit workflow

        This tool uses progressive disclosure: it shows only statistics (additions/deletions per file)
        to avoid token limits. Use list_changes or diff tools to view detailed changes for specific files.

        Use this when you have multiple logical changes mixed together and want to organize them
        into separate, well-structured commits.

        Returns:
            JSON with recent commits, file statistics, and next steps for the agent
        """
        # Get recent non-merge commits (last 5)
        try:
            log_output = run([
                "git", "log",
                "--no-merges",
                "--pretty=format:%s",
                "-5"
            ])
            recent_commits = [line.strip() for line in log_output.strip().split("\n") if line.strip()]
        except Exception as e:
            recent_commits = [f"Error getting commits: {str(e)}"]

        # Get all unstaged changes with statistics only (progressive disclosure)
        diff_text, untracked_set, deleted_set = get_diff_with_untracked([], UNIFIED_LIST_DEFAULT)
        files_hunks, binaries = parse_unified_diff(diff_text)

        # Build file statistics without including full diff content
        file_stats = []
        total_additions = 0
        total_deletions = 0

        for path in sorted(files_hunks.keys()):
            hunks = files_hunks[path]
            binflag = binaries.get(path, False)

            # Determine file status
            if path in untracked_set:
                status = "added"
            elif path in deleted_set:
                status = "deleted"
            else:
                status = "modified"

            if binflag:
                file_stats.append({
                    "path": path,
                    "binary": True,
                    "status": status,
                    "additions": 0,
                    "deletions": 0,
                    "changes": 0
                })
            else:
                stats = calculate_line_stats(hunks)
                file_stats.append({
                    "path": path,
                    "binary": False,
                    "status": status,
                    "additions": stats["additions"],
                    "deletions": stats["deletions"],
                    "changes": stats["changes"]
                })
                total_additions += stats["additions"]
                total_deletions += stats["deletions"]

        # Create instruction for the agent
        instruction = {
            "recent_commits": recent_commits,
            "summary": {
                "total_files": len(file_stats),
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "total_changes": total_additions + total_deletions
            },
            "files": file_stats,
            "next": (
                "Now follow these steps to create focused commits:\n\n"
                "1. **Review file statistics**: Above you see all changed files with line counts "
                "(additions/deletions). Use list_changes or diff tools to view detailed changes for specific files.\n\n"
                "2. **Analyze the changes**: Identify logical groups that should be committed together "
                "(e.g., related bug fixes, new features, refactoring).\n\n"
                "3. **Plan commits**: Decide how many commits you need and what each should contain. "
                "Use the recent commit messages above as a style reference.\n\n"
                "4. **Use TodoWrite**: If you can manage todos, create a todo item for each planned commit "
                "with its intended commit message. Mark them as pending.\n\n"
                "5. **Stage changes**: For each commit:\n"
                "   - Mark the todo as in_progress\n"
                "   - Use list_changes to view the numbered changes for relevant files\n"
                "   - Use apply_changes to stage the relevant line numbers\n"
                "   - Run `git commit -m \"your message\"` to create the commit\n"
                "   - Mark the todo as completed\n\n"
                "6. **Verify**: After all commits, run `git log` to verify and `git status` to ensure "
                "no changes were missed.\n\n"
                "Remember: Create focused, atomic commits. Each commit should represent one logical change."
            )
        }

        return json.dumps(instruction, ensure_ascii=False, indent=2)

    @mcp.prompt(name="auto-commit", description="Organize and commit changes using git-polite tools.")
    def auto_commit_command() -> str:
        """Organize and commit changes using git-polite tools."""
        return (
            "Call auto_commit and follow the instructions it returns."
        )

    @mcp.tool(annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        openWorldHint=True
    ))
    def unstack(branches: dict[str, list[str]], parent: str = "origin/main") -> str:
        """Unstack linear commits into parallel branches for separate PRs.

        This tool transforms a linear commit history (A -> B -> C -> D) into parallel
        branches (A -> B, A -> C, A -> D) where each branch contains specific commits
        cherry-picked from the original history.

        Use this when you've made multiple changes in sequence but want to create
        separate PRs for different logical changes. Each branch can be independently
        reviewed and merged.

        Example scenario:
        You have commits: fix-bug -> add-feature -> update-docs
        You want separate PRs, so you create:
        - feat/999: [fix-bug, update-docs]
        - feat/1000: [add-feature]

        This creates two branches from origin/main:
        - feat/999 with fix-bug and update-docs cherry-picked in order
        - feat/1000 with add-feature cherry-picked

        Args:
            branches: Dictionary mapping branch names to lists of commit references.
                     Commits can be specified as SHA, branch names, or symbolic refs (e.g., HEAD~2).
                     Commits are cherry-picked in the order specified.
            parent: Base commit to branch from (default: "origin/main").
                   All branches will start from this commit.

        Returns:
            JSON string with format: {
                created_branches: [{name, commits_applied, head_sha}],
                errors: [{branch, commit, error}],
                stats: {total_branches, successful_branches, failed_branches}
            }

        Note:
            - Existing branches with the same name will cause an error
            - The current branch is not changed by this operation
            - Uses low-level git commands (commit-tree, update-ref) to avoid changing working directory
        """
        result = do_unstack(branches, parent)
        return json.dumps(result, ensure_ascii=False, indent=2)

    return mcp

def main():
    args = parse_args()
    if args.cmd == "list":
        resp = list_files(args.paths, args.page_token, args.page_size_files, args.page_size_bytes, args.unified)
        if args.format == "pretty":
            print(format_pretty(resp))
        else:
            json.dump(resp, sys.stdout, ensure_ascii=False)
            sys.stdout.write("\n")
        return

    if args.cmd == "apply":
        try:
            numbers = parse_number_tokens(args.numbers)
        except ValueError as e:
            if args.format == "pretty":
                print(f"{ANSI_RED}Error: {e}{ANSI_RESET}", file=sys.stderr)
            else:
                print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(2)
        resp = apply_one_file(args.path, numbers)
        if args.format == "pretty":
            print(format_apply_pretty(resp))
        else:
            json.dump(resp, sys.stdout, ensure_ascii=False)
            sys.stdout.write("\n")
        return

    if args.cmd == "mcp":
        mcp = create_mcp_server()
        mcp.run()
        return

if __name__ == "__main__":
    main()

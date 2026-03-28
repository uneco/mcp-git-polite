<div align="center">
 <img src="https://github.com/user-attachments/assets/1a84d7d4-7393-438b-92ef-55c3edac9461" width="1500" height="auto" alt="mcp-git-polite"/>
</div>

<hr />

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/uneco/mcp-git-polite/test.yml?branch=main)](https://github.com/uneco/mcp-git-polite/actions)
[![GitHub](https://img.shields.io/github/license/uneco/mcp-git-polite)](https://github.com/uneco/mcp-git-polite/blob/main/LICENSE)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/uneco/mcp-git-polite)](https://github.com/uneco/mcp-git-polite/pulse)
[![GitHub last commit](https://img.shields.io/github/last-commit/uneco/mcp-git-polite)](https://github.com/uneco/mcp-git-polite/commits/main)

Git staging on autopilot — let AI organize your changes into clean, focused commits.

## Overview

`git-polite` is a Model Context Protocol (MCP) server that brings intelligent git staging to AI agents. It can automatically organize messy work-in-progress into well-structured commits, or give you surgical precision with line-by-line staging when you need it.

## Features

- **Autopilot Mode**: Let AI analyze your changes and create multiple focused commits automatically
- **Line-Level Staging**: Stage individual additions and deletions by line number with surgical precision
- **Untracked File Support**: Stage parts of newly created files (not just modified files)
- **Range Selection**: Apply multiple changes at once using ranges (e.g., `0001-0005,0020-0025`)
- **LLM-Friendly Output**: Byte-based pagination and smart truncation protect context windows
- **Binary File Detection**: Automatically detects and skips binary files
- **MCP Integration**: Works seamlessly with Claude Code, Claude Desktop, and other MCP clients

### MCP Server Mode

Run as an MCP server for integration with MCP clients:

```bash
uv run git_polite.py mcp
```

#### MCP Tools

The server exposes four tools:

1. **list_changes**: List unstaged git changes (including untracked files) as numbered lines
   - Smart truncation: Large diffs (>10KB) are automatically truncated to protect LLM context
   - Parameters:
     - `paths` (optional): List of file paths to filter
     - `page_token` (optional): Pagination token
     - `page_size_files` (optional, default: 50): Max files per page
     - `page_size_bytes` (optional, default: 30KB): Max cumulative bytes per page
     - `unified` (optional, default: 20): Context lines around changes
   - Output includes `truncated: true` flag for large files with a `reason` explaining the truncation
   - For truncated files, use the `diff` tool to view complete content

2. **diff**: View complete diff for a single file without truncation
   - Use this for files that are truncated in `list_changes` output
   - Returns the same numbered line format as `list_changes`, enabling partial staging
   - Unlike `git diff`, this tool provides line numbers required by `apply_changes`
   - Never truncates output, suitable for large files with extensive changes
   - Parameters:
     - `path` (required): File path to view diff for
     - `unified` (optional, default: 20): Context lines around changes
   - Returns: Complete diff with `size_bytes` indicating actual output size

3. **apply_changes**: Apply selected changes to git index by number (supports partial staging of untracked files)
   - Parameters:
     - `path`: File path to apply changes to
     - `lines`: Change numbers (format: `NNNN,MMMM,PPPP-QQQQ`)

4. **auto_commit**: Start autopilot mode to organize all changes into focused commits
   - Shows recent commit messages for style reference
   - Analyzes all unstaged changes and suggests logical groupings
   - Guides AI through creating multiple atomic commits from messy WIP

### MCP Client Configuration

#### Using uvx (Recommended)

Use `uvx` to run directly from GitHub:

**Claude Desktop Configuration:**

```json
{
  "mcpServers": {
    "git-polite": {
      "command": "uvx",
      "args": [
        "git-polite@git+https://github.com/uneco/mcp-git-polite.git",
        "mcp"
      ]
    }
  }
}
```

**Claude CLI:**

```bash
claude mcp add -s user git-polite uvx git-polite@git+https://github.com/uneco/mcp-git-polite.git mcp
```

#### Using Docker

Alternatively, use the Docker image from GitHub Container Registry:

```json
{
  "mcpServers": {
    "git-polite": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v",
        "${workspaceFolder}:/workspace",
        "-w",
        "/workspace",
        "ghcr.io/uneco/mcp-git-polite:latest",
        "mcp"
      ]
    }
  }
}
```

## How It Works

1. **List Phase**: The tool parses `git diff` output (including untracked files via `git diff --no-index`) and numbers each addition (`+`) and deletion (`-`) sequentially
2. **Truncation Check**: Each file's diff size is measured. Files exceeding 10KB are marked as truncated to protect LLM context
3. **Display**: Changes are shown with their numbers, along with surrounding context lines. Files are marked as "added" (untracked), "modified", or "deleted"
4. **Pagination**: Results are paginated based on cumulative byte size (default 30KB per page) to prevent overwhelming the LLM
5. **Apply Phase**: When you specify line numbers, the tool:
   - Reads the staged version from git index (or creates new file for untracked files)
   - Applies only the selected changes
   - Updates the git index with the partial changes

## Output Format

### List Output

```json
{
  "page_token_next": "optional-token",
  "files": [
    {
      "path": "src/main.py",
      "binary": false,
      "status": "modified",
      "lines": [
        "0001: + new line 1",
        "        context line",
        "0002: - deleted line",
        "0003: + new line 2",
        "        ..."
      ]
    },
    {
      "path": "src/new_file.py",
      "binary": false,
      "status": "added",
      "lines": [
        "0001: + def hello():",
        "0002: +     print('Hello')"
      ]
    },
    {
      "path": "src/refactored_module.py",
      "binary": false,
      "status": "modified",
      "truncated": true,
      "reason": "diff too large (45.2 KB, max 10 KB)",
      "lines": []
    }
  ],
  "stats": {
    "files": 3,
    "lines": 5,
    "truncated_files": 1,
    "page_bytes": 4532
  }
}
```

When a file shows `truncated: true`, use the `diff` tool to view its complete content. The `diff` tool provides the same numbered line format needed for partial staging, which `git diff` cannot provide.

### Apply Output

```json
{
  "applied": [
    {
      "file": "src/main.py",
      "applied_count": 3,
      "after_applying": {
        "diff": ["0001: + remaining", "0002: - unstaged", "0003: + changes"],
        "unstaged_lines": 5
      }
    }
  ],
  "skipped": [],
  "stats": {
    "files": 1,
    "changes_applied": 3,
    "changes_skipped": 0
  }
}
```

## Requirements

- Python 3.10 or higher
- Git (command-line tool)
- MCP server package (`mcp>=1.10.0`)

## Development

```bash
# Install dependencies
uv sync

# Run tests (if available)
uv run pytest

# Format code
uv run black git_polite.py

# Type check
uv run mypy git_polite.py
```

## Use Cases

- **AI-Powered Commit Organization**: Let AI analyze your WIP and create clean commit history automatically
- **Incremental Commits**: Break down large changes into logical, atomic commits
- **Partial File Staging**: Stage only specific lines of a new file while keeping the rest unstaged
- **Code Review Preparation**: Stage related changes together, even if scattered across files
- **Refactoring**: Separate formatting changes from logic changes with surgical precision

## Example Workflows

### Working with Truncated Files

When you encounter a truncated file (e.g., a large refactored file with many changes):

```python
# Step 1: List all changes
result = list_changes()

# Step 2: Notice a truncated file
# {
#   "path": "src/api_client.py",
#   "truncated": true,
#   "reason": "diff too large (45.2 KB, max 10 KB)",
#   "lines": []
# }

# Step 3: View the complete numbered diff
# Use the diff tool (not git diff) because it provides line numbers needed for partial staging
full_diff = diff(path="src/api_client.py")

# Step 4: Selectively stage related changes (e.g., bug fixes separate from refactoring)
apply_changes(path="src/api_client.py", lines="0001-0050,0120-0135")
```

### Pagination Example

```python
# Get first page (max 30KB)
page1 = list_changes(page_size_bytes=30720)

# Continue with next page if needed
if page1["page_token_next"]:
    page2 = list_changes(page_token=page1["page_token_next"])
```

## Limitations

- Works only with text files (binary files are detected and skipped)
- Line numbers are ephemeral - they change after each apply operation
- Context mismatches (file drift) will cause operations to fail safely
- For untracked files, the entire file content must be present in the working directory
- Large files (>10KB diff) are truncated in `list_changes` - use `diff` tool to view them

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built with [FastMCP](https://gofastmcp.com/) and the [Model Context Protocol](https://modelcontextprotocol.io/).

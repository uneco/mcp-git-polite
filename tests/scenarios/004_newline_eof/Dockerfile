# Scenario: Newline EOF - Repository with files testing trailing newline handling
# Initial state:
#   - File without trailing newline (committed, then modified)
#   - File with trailing newline (committed, then modified)
#   - File where trailing newline is removed
#   - File where trailing newline is added
# Use for: Testing apply_changes with trailing newline edge cases

FROM git-polite-test-base

# Create template repository (will be copied to /repo before each test)
WORKDIR /repo_template

# Initialize repository
RUN git init

# Case 1: File without trailing newline (committed, then modified without trailing newline)
RUN printf "line 1\nline 2\nline 3" > no_newline.txt && \
    git add no_newline.txt && \
    git commit -m "Add file without trailing newline"

# Modify it (still no trailing newline)
RUN printf "line 1\nmodified line 2\nline 3\nadded line 4" > no_newline.txt

# Case 2: File with trailing newline (committed, then modified with trailing newline)
RUN printf "line 1\nline 2\nline 3\n" > with_newline.txt && \
    git add with_newline.txt && \
    git commit -m "Add file with trailing newline"

# Modify it (keep trailing newline)
RUN printf "line 1\nmodified line 2\nline 3\nadded line 4\n" > with_newline.txt

# Case 3: File where trailing newline is removed
RUN printf "line 1\nline 2\n" > remove_newline.txt && \
    git add remove_newline.txt && \
    git commit -m "Add file with trailing newline for removal test"

# Remove trailing newline
RUN printf "line 1\nline 2" > remove_newline.txt

# Case 4: File where trailing newline is added
RUN printf "line 1\nline 2" > add_newline.txt && \
    git add add_newline.txt && \
    git commit -m "Add file without trailing newline for addition test"

# Add trailing newline
RUN printf "line 1\nline 2\n" > add_newline.txt

# Set working directory for pytest (not /repo!)
WORKDIR /workspace

# Run pytest (fixture will copy /repo_template to /repo before each test)
CMD ["pytest", "-v", "-p", "no:cacheprovider", "-m", "newline_eof", "/tests/"]

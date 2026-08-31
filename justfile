# Setup the repo .venv via uv
setup:
    uv sync

# Run the test suite
test:
    uv run pytest

# Run static analysis and automatically fix issues where possible
check:
    uvx ruff check . --fix

# Format code according to project style
format:
    uvx ruff format .

# Run formatting and linting (CI-style target)
clean: format check

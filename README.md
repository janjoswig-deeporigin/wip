# WIP

A minimal Python project structure for ad-hoc and developement work.

## Installation

```bash
pip install -e .
```

## Development

Install in development mode with optional dependencies:

```bash
pip install -e ".[dev]"
```

## Testing

Run tests with:

```bash
pytest
```

## Code Quality

Format code with:

```bash
ruff format
ruff check --select I --fix
```

Check code quality with:

```bash
mypy src
```
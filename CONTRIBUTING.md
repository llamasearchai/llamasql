# Contributing to LlamaDB

We love your input! We want to make contributing to LlamaDB as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

### Code Style

We use several tools to ensure consistent code style:

- **Python**: We follow PEP 8 and use `black` for auto-formatting, `ruff` for linting, and `mypy` for type checking
- **Rust**: We follow standard Rust style and use `rustfmt` for formatting and `clippy` for linting

To ensure your code passes these checks, run:

```bash
# Format Python code
black python/ tests/

# Lint Python code
ruff check python/ tests/

# Type check Python code
mypy python/

# Format Rust code
cd rust_extensions/ && cargo fmt

# Lint Rust code
cd rust_extensions/ && cargo clippy
```

## Development Environment Setup

### Prerequisites

- Python 3.11+
- Rust (stable channel)
- PostgreSQL (optional, for full database testing)

### Setting Up

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/llamadb.git
cd llamadb

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Build Rust extensions
maturin develop
```

## Running Tests

Before submitting a pull request, please make sure all tests pass:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=llamadb tests/

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

## Branching Strategy

- `main` - Main development branch, should always be stable
- `feature/*` - Feature branches
- `bugfix/*` - Bug fix branches
- `release/*` - Release preparation branches

## Commit Messages

We follow conventional commits for our commit messages:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

Example: `feat(api): add new endpoint for vector search`

## Issue Reporting Guidelines

When reporting issues, please use one of our templates and include as much detail as possible:

- For bugs: Include steps to reproduce, expected behavior, actual behavior, and your environment details
- For features: Include a clear description of the feature, the motivation for it, and any potential implementation ideas

## Roadmap and Project Management

We maintain a [Roadmap](docs/ROADMAP.md) document that outlines our planned features and enhancements. For more detailed tracking, we use GitHub Projects.

## Licensing

By contributing to LlamaDB, you agree that your contributions will be licensed under the same [Apache License 2.0](LICENSE) that covers the project.

## Code of Conduct

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to make participation in our project and our community a harassment-free experience for everyone. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## Questions?

If you have any questions, feel free to reach out to the maintainers:

- Open a discussion on GitHub
- Join our community on Discord (link in README)

Thank you for contributing to LlamaDB! 
# Contributing Guidelines

Thank you for considering contributing to this project.

## How to Contribute

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/your-username/Feature-Selection-Tool.git`
3. Create a branch: `git checkout -b feature/your-feature` or `bugfix/issue-description`
4. Install in editable mode: `pip install -e ".[dev]"`
5. Make your changes and run tests: `pytest tests/ -q`
6. Commit: `git commit -m "Your descriptive commit message"`
7. Push: `git push origin feature/your-feature`
8. Open a pull request.

## Bug Reports

Open an issue with steps to reproduce, expected vs actual behavior, and environment details.

## Feature Requests

Open an issue describing the use case and proposed behavior.

## Style Guide

- Prefer clear, small functions and sklearn-style `fit` / `transform` APIs.
- Keep train-only fitting for any preprocessor or selector (no data leakage).
- Do not impute target labels; drop missing labels instead.
- New selection methods should plug into `FeatureSelector` via `normalize_method` and `_fit_*` helpers, and be covered in `tests/test_methods.py` plus a compare smoke test.

## License

By contributing, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE).

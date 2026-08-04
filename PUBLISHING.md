# Publishing to PyPI

Package name on PyPI: **`feature-selector-tool`**  
Import name: **`feature_selector`**  
CLI: **`feature-select`**

## Local build (always do this first)

```bash
pip install -e ".[dev]"
pytest tests/ -q
python -m build
twine check dist/*
```

Inspect the wheel:

```bash
python -m pip install dist/feature_selector_tool-*.whl --force-reinstall
feature-select --version
python -c "from feature_selector import FeatureSelector; print(FeatureSelector)"
```

## TestPyPI (recommended dry run)

1. Create an account at https://test.pypi.org/  
2. Create an API token.  
3. Upload:

```bash
twine upload --repository testpypi dist/*
```

4. Install from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple \
  feature-selector-tool
```

## Production PyPI

```bash
# clean previous builds
rm -rf dist/ build/ *.egg-info

python -m build
twine check dist/*
twine upload dist/*
```

Requires a PyPI account and API token (`TWINE_USERNAME=__token__`, `TWINE_PASSWORD=pypi-...`).

## GitHub Release automation

This repo includes `.github/workflows/publish.yml`:

- Triggers on a **published GitHub Release**
- Builds the wheel/sdist
- Uploads to PyPI if `PYPI_API_TOKEN` is set as a repository secret

Steps:

1. Bump version in `pyproject.toml` and `feature_selector/__init__.py`
2. Update `CHANGELOG.md`
3. Push to `main`
4. Create a GitHub Release tagged `v0.5.0` (matching the version)
5. Ensure secret `PYPI_API_TOKEN` exists under repo Settings → Secrets

## Version policy

- **0.x** = research-grade beta (API may evolve carefully)
- Bump **minor** for features (0.4 → 0.5)
- Bump **patch** for fixes only

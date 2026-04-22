# Releasing adaptive-lasso

This repository is configured for Trusted Publishing with GitHub Actions.

The package version is single-sourced from:

- `src/adaptive_lasso/__init__.py`

The current workflow file for publishing is:

- `.github/workflows/release.yml`

## Versioning policy

This project uses PEP 440-compatible versions and follows a semver-like public
release policy:

- `1.0.0`: first public package release
- `1.0.1`: bugfix or packaging-only release
- `1.1.0`: backwards-compatible feature release
- `2.0.0`: intentional breaking API change

Pre-releases are also valid when needed, for example:

- `1.1.0rc1`
- `1.2.0.dev1`

## One-time setup

### 1. Configure GitHub environments

In GitHub repository settings, create these environments:

- `testpypi`
- `pypi`

Optional but recommended:

- require manual approval for `pypi`
- restrict who can approve production releases

### 2. Configure Trusted Publishers

You need to configure Trusted Publishers separately on TestPyPI and PyPI.

For GitHub Actions, use:

- owner: `AaronNHorvitz`
- repository: `AdaptiveLASSO`
- workflow: `.github/workflows/release.yml`
- environment: `testpypi` or `pypi`

If the project does not exist yet on the index, create a pending publisher for:

- project name: `adaptive-lasso`

Official docs:

- PyPI pending publishers:
  https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
- Existing projects:
  https://docs.pypi.org/trusted-publishers/adding-a-publisher/

## Release checklist

### 1. Bump the version

Edit:

- `src/adaptive_lasso/__init__.py`

Example:

```python
__version__ = "1.0.1"
```

### 2. Verify locally

Run:

```bash
.venv/bin/pip install -e .[dev]
.venv/bin/pytest
.venv/bin/python -m build
.venv/bin/twine check dist/*
```

### 3. Commit and push

```bash
git add src/adaptive_lasso/__init__.py pyproject.toml RELEASING.md .github/workflows/release.yml
git commit -m "Prepare v1.0.1 release"
git push origin master
```

### 4. Publish to TestPyPI first

In GitHub Actions:

- run the `release` workflow manually
- choose `repository = testpypi`

After it completes, test installation from TestPyPI in a fresh virtualenv:

```bash
python -m venv /tmp/adaptive-lasso-test
/tmp/adaptive-lasso-test/bin/pip install --upgrade pip
/tmp/adaptive-lasso-test/bin/pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ adaptive-lasso==1.0.1
```

### 5. Tag the release

```bash
git tag v1.0.1
git push origin v1.0.1
```

### 6. Publish to PyPI

Preferred path:

- create a GitHub Release from tag `v1.0.1`
- publish the release

That triggers `.github/workflows/release.yml`, which:

- runs tests
- builds the wheel and sdist
- checks the distributions with `twine check`
- publishes to PyPI via Trusted Publishing

You can also manually dispatch the workflow with `repository = pypi`.

## Notes

- PyPI will not let you reuse the same uploaded version/file name. If a release
  needs to be fixed, bump the version and publish again.
- If a release should no longer be recommended, prefer yanking it instead of
  deleting it.

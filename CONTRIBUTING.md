# Contributing

Thank you for considering contributing to SHEAP!

Contributions are welcome. This guide explains how to report bugs, propose features, improve documentation, set up the project locally, run tests, and build the package using `uv`.

* [Types of Contributions](#types-of-contributions)
* [Contributor Setup](#setting-up-the-code-for-local-development)
* [Contributor Guidelines](#contributor-guidelines)
* [Contributor Testing](#testing-with-tox)
* [Core Committer Guide](#core-committer-guide)

## Types of Contributions

You can contribute in many ways.

### Report Bugs

Report bugs at https://github.com/favila/sheap/issues.

A bug means behavior that differs from the expected or documented behavior. When reporting a bug, please include the following information by filling in the bug report template:

* Your operating system name and version.
* Any details about your local setup that may help with troubleshooting.
* Detailed steps to reproduce the bug, if possible.
* If you do not have exact reproduction steps, describe your observations as clearly as possible.

Questions that help start a discussion about the issue are also welcome.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with `bug` is open to contributors who want to implement a fix.

See [Contributor Setup](#setting-up-the-code-for-local-development) to get started.

### Implement Features

Look through the GitHub issues for feature requests. Anything tagged with `enhancement` or `please-help` is open to contributors.

Please do not combine multiple unrelated feature enhancements into a single pull request. Smaller, focused pull requests are easier to review and maintain.

See [Contributor Setup](#setting-up-the-code-for-local-development) to get started.

### Write Documentation

SHEAP can always use more documentation, whether in the official documentation, examples, tutorials, or docstrings.

To build and preview the documentation locally, use:

```bash
uv sync --extra docs
uv run tox -e live-html
```

This compiles the documentation into HTML and watches the files for changes, recompiling when you save. You can open the local documentation in your browser at:

```text
http://127.0.0.1:8000
```

### Submit Feedback

The best way to send feedback is to open an issue at:

```text
https://github.com/favila/sheap/issues
```

If you are proposing a feature:

* Explain clearly how it should work.
* Keep the scope as narrow as possible.
* Remember that this is a volunteer-driven project and contributions are welcome.

## Setting Up the Code for Local Development

This project uses `uv` for local development. `uv` manages the project environment, dependencies, lockfile, and command execution.

### 1. Install `uv`

If you do not already have `uv` installed, install it following the official instructions:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, check that `uv` is available:

```bash
uv --version
```

### 2. Fork and clone the repository

Fork the `sheap` repository on GitHub, then clone your fork locally:

```bash
git clone git@github.com:favila/sheap.git
cd sheap
```

### 3. Install the required Python version

If the project defines a `.python-version` file, `uv` can use it automatically. You can also install the required Python version explicitly.

For example, if SHEAP requires Python 3.12:

```bash
uv python install 3.12
```

Then create or update the local `.python-version` file if needed:

```bash
uv python pin 3.12
```

### 4. Create and sync the development environment

Install the project and its dependencies into the local `uv` environment:

```bash
uv sync
```

If you want to install all optional development dependencies, use:

```bash
uv sync --all-extras
```

If the project defines dependency groups such as `dev`, `docs`, or `test`, you can sync them with:

```bash
uv sync --group dev
uv sync --group docs
uv sync --group test
```

or combine them:

```bash
uv sync --group dev --group docs --group test
```

### 5. Activate the environment, optional

You do not need to activate the environment if you use `uv run`.

For example:

```bash
uv run python --version
uv run pytest
```

If you prefer activating the environment manually:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 6. Create a branch for local development

Create a branch for your bug fix or feature:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

## Contributor Guidelines

### Pull Request Guidelines

Before submitting a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. The pull request should be focused. If it is too large, consider splitting it into smaller pull requests.
3. If the pull request adds or changes functionality, the documentation should be updated.
4. The pull request must pass all CI/CD jobs before it is ready for review.
5. If a CI/CD job is failing for unrelated reasons, consider opening a separate pull request to fix that first.

### Coding Standards

SHEAP follows these general design principles:

* Single responsibility of units.
* Modularity.
* Composition over inheritance.
* Clear APIs.
* Maintainable and tested code.

## Testing with tox

SHEAP uses `tox` to run the test suite and other quality checks.

You can run `tox` through `uv` without installing it globally:

```bash
uv run tox
```

If the project requires the `PKG_VERSION` environment variable, use:

```bash
PKG_VERSION=$(uv run python ./scripts/parse_version.py) uv run tox
```

To avoid repeating this command, you can define a shell alias:

```bash
alias tox-sheap='PKG_VERSION=$(uv run python ./scripts/parse_version.py) uv run tox'
```

Then run:

```bash
tox-sheap
```

Please note that `tox` runs the test suite against multiple Python versions if those versions are available on the host machine.

### Run all tests

```bash
uv run tox
```

### Run tests for a specific environment

For example, to run the Python 3.12 environment:

```bash
uv run tox -e py312
```

### Run only tests matching a pattern

For example, to run tests matching `smoke_test`:

```bash
uv run tox -e py312 -- -k "smoke_test"
```

### Run pytest directly

For quick local checks, you can also run `pytest` directly:

```bash
uv run pytest
```

To run a specific test file:

```bash
uv run pytest tests/test_example.py
```

To run a specific test:

```bash
uv run pytest tests/test_example.py::test_name
```

## Building the Package

To build the source distribution and wheel using `uv`, run:

```bash
uv build
```

This creates the package distributions inside the `dist/` directory.

To check the generated distributions, run:

```bash
uvx twine check dist/*
```

You can also remove previous build artifacts before rebuilding:

```bash
rm -rf dist build *.egg-info
uv build
uvx twine check dist/*
```

## Documentation Checks

To build the documentation locally, use:

```bash
uv sync --extra docs
uv run tox -e docs
```

If the project has a live documentation environment:

```bash
uv run tox -e live-html
```

## Before Committing

Before committing your changes, run the relevant checks:

```bash
uv sync --all-extras
uv run tox
uv build
uvx twine check dist/*
```

Check the coverage report printed in the console when running tests.

If an HTML coverage report is generated, it will usually be placed in:

```text
htmlcov/
```

Do not commit the coverage report directory.

## Commit and Push

When your changes are ready:

```bash
git add -p
git commit -m "Your detailed description of your changes"
git push origin name-of-your-bugfix-or-feature
```

Then submit a pull request through the GitHub website.

## Core Committer Guide

### Vision and Scope

Core committers should use this section to:

* Guide decisions as maintainers.
* Keep the project focused.
* Avoid unnecessary long-term maintenance burden.

### API Accessible

SHEAP should provide:

* A modular API that strives for statelessness.
* A simple interface for common use cases.
* Flexibility for more complex workflows.
* Extensibility for advanced users.

### Extensible

The codebase should prioritize:

* Modular design.
* Stateless components when possible.
* Clear separation between data structures, fitting logic, plotting, and utilities.

### Fast and Focused

SHEAP is designed to do one thing well: spectral handling and estimation of AGN parameters.

The project should cover the most important use cases while avoiding unnecessary complexity.

### Inclusive

SHEAP should aim for:

* Cross-platform support.
* Clear documentation.
* Reproducible installation.
* Welcoming contribution practices.

### Stable

SHEAP should prioritize:

* High test coverage.
* Tests for corner cases.
* Stable APIs that users and tool builders can rely on.
* No pull requests that reduce test coverage without justification.

## Process: Pull Requests

Prioritize pull requests in this order:

1. Fixes for broken tests on any supported platform or Python version.
2. Additional tests covering corner cases.
3. Minor documentation edits.
4. Bug fixes.
5. Major documentation improvements.
6. Features.

### Pull Request Review Guidelines

When reviewing pull requests:

* Think carefully about the long-term implications of the change.
* Consider how the change affects existing projects that depend on SHEAP.
* Be strict about quality, maintainability, and tests.
* Ask for improvements when needed.
* When merging a pull request, close or update every related issue and explain how it was affected.
* Remember to add the contributor to `AUTHORS.md` when appropriate.

## Process: Issues

If an issue is an urgent bug, mark it for the next patch release.

Then either:

* Fix it directly, or
* Mark it as `please-help`.

For other issues:

* Encourage friendly discussion.
* Moderate debate constructively.
* Offer technical guidance when useful.

## Process: Roadmap

The roadmap is available at:

```text
https://github.com/favila/sheap/milestones?direction=desc&sort=due_date&state=open
```

Due dates are flexible.

## Process: Release

SHEAP follows semantic versioning.

See:

```text
https://semver.org
```

A typical release workflow is:

```bash
git status
uv sync --all-extras
uv run tox
rm -rf dist build *.egg-info
uv build
uvx twine check dist/*
```

Then create and push a version tag:

```bash
git tag v0.0.1
git push origin v0.0.1
```

If a tag was created by mistake, delete it locally and remotely:

```bash
git tag -d v0.0.1
git push origin --delete v0.0.1
```

Package publishing should be handled only by maintainers or through the configured CI/CD release workflow.

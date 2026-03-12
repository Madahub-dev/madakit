# Release Checklist

This document outlines the steps for releasing a new version of madakit.

## Pre-Release

- [x] All tests passing (`pytest`)
- [x] Linting passes (`ruff check .`)
- [x] Type checking passes (`mypy src/`)
- [x] Documentation is up-to-date
- [x] CHANGELOG.md updated with release notes
- [x] Version bumped in `pyproject.toml`
- [x] `__version__` added/updated in `src/madakit/__init__.py`
- [x] Development Status classifier updated (if major version)

## Release Process

### 1. Version Bump (Task 17.5.1)

- [x] Update version in `pyproject.toml`
- [x] Add/update `__version__` in `src/madakit/__init__.py`
- [x] Update `Development Status` classifier if needed
- [x] Commit changes: `git commit -m "chore(release): bump version to X.Y.Z"`

### 2. Git Tag (Task 17.5.2)

- [x] Create annotated tag:
  ```bash
  git tag -a vX.Y.Z -m "Release vX.Y.Z

  Brief description of major features/changes.
  "
  ```
- [x] Verify tag: `git show vX.Y.Z --no-patch`
- [ ] Push tag to origin: `git push origin vX.Y.Z`

### 3. Build and Publish (Task 17.5.3)

#### Option A: Manual Publishing

1. Build distribution packages:
   ```bash
   ./scripts/build.sh
   ```

2. Publish to TestPyPI (optional):
   ```bash
   twine upload --repository testpypi dist/*
   ```

3. Test installation from TestPyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ madakit
   ```

4. Publish to PyPI:
   ```bash
   twine upload dist/*
   ```

5. Verify on PyPI: https://pypi.org/project/madakit/

#### Option B: GitHub Actions (Recommended)

1. Push tag to GitHub:
   ```bash
   git push origin vX.Y.Z
   ```

2. GitHub Actions will automatically:
   - Build distribution packages
   - Run tests
   - Publish to PyPI (if tag matches `v*.*.*`)
   - Create GitHub release with CHANGELOG excerpt
   - Attach distribution files to release

3. Verify:
   - Check GitHub Actions workflow: https://github.com/Madahub-dev/madakit/actions
   - Check PyPI: https://pypi.org/project/madakit/
   - Check GitHub release: https://github.com/Madahub-dev/madakit/releases

### 4. Announcement (Task 17.5.4)

- [x] Create release notes (`RELEASE_NOTES.md`)
- [ ] Update GitHub release with full release notes
- [ ] Announce on social media (Twitter, Reddit, etc.)
- [ ] Post on relevant communities (Python Discord, Hacker News, etc.)
- [ ] Update documentation site (if manual deployment needed)
- [ ] Send announcement to mailing list (if applicable)

## Post-Release

- [ ] Verify package on PyPI
- [ ] Test installation: `pip install madakit==X.Y.Z`
- [ ] Test import: `python -c "import madakit; print(madakit.__version__)"`
- [ ] Monitor GitHub issues for release-related bugs
- [ ] Update project roadmap
- [ ] Begin next development cycle

## Release Candidate (RC) Process

For major releases, consider creating release candidates first:

1. Create RC tag: `vX.Y.Z-rc1`
2. Publish to TestPyPI only
3. Gather feedback from early testers
4. Fix any issues and create `vX.Y.Z-rc2` if needed
5. Once stable, create final `vX.Y.Z` tag

## Hotfix Process

For urgent bug fixes on a released version:

1. Create hotfix branch from release tag: `git checkout -b hotfix/X.Y.Z+1 vX.Y.Z`
2. Fix the bug and commit
3. Update version to `X.Y.Z+1`
4. Update CHANGELOG.md with hotfix notes
5. Create tag `vX.Y.Z+1`
6. Merge back to main and dev branches
7. Follow normal release process

## Rollback Process

If a release has critical issues:

1. Yank the release on PyPI (doesn't delete, but warns users):
   ```bash
   # Via PyPI web interface or twine
   ```

2. Create hotfix release with fix
3. Announce the issue and recommend upgrade

## Version Numbering

madakit follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0) — Breaking changes
- **MINOR** (0.X.0) — New features, backward compatible
- **PATCH** (0.0.X) — Bug fixes, backward compatible

## Current Release: v1.0.0

- [x] Task 17.5.1: Version bump complete
- [x] Task 17.5.2: Git tag created
- [ ] Task 17.5.3: PyPI publish pending
- [ ] Task 17.5.4: Announcement pending

**Status:** Ready for PyPI publishing and announcement.

**Next steps:**
1. Push tag to GitHub: `git push origin v1.0.0`
2. Monitor GitHub Actions for automated release
3. Publish release notes and announce

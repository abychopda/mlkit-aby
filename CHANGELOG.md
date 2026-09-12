# Changelog

<!--next-version-placeholder-->

## v0.2.0 (12/09/2026)

This release renames and generalises the package so that it can be reused
across projects rather than being tied to a single assignment dataset.

### Changed (breaking)

- Renamed the distribution from `36120-26sp-group36-25611094-package` to
  `mlkit-aby`, and the import module from `group36_25611094_package` to
  `mlkit_aby`.

### Removed (breaking)

- `helpers.cleaning.height_to_inches` — specific to the basketball dataset
  used in the previous project.
- `helpers.cleaning.calculate_age` — specific to the same dataset.
- `helpers.submission` — specific to Kaggle competition submissions.

### Added

- Project hygiene: a `.gitignore` covering Python caches, build artefacts
  and virtual environments, and removal of previously committed
  `__pycache__` directories.

## v0.1.0 (20/08/2026)

- First release, as `36120_26sp_group36_25611094_package`.

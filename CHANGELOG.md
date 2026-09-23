# Changelog

<!--next-version-placeholder-->

## v0.3.1 (23/09/2026)

### Added

- `helpers.features.add_rain_features` — rainfall features that vary more
  usefully than raw millimetres, which are zero on about half of all days:
  a wet day flag, rolling counts of rain days, rolling rainfall totals and
  the length of the current dry spell. Enabled by the `rain_features` key
  in a recipe, and included in `DEFAULT_RECIPE`.
- `required_history_days` now accounts for the rainfall windows, so a recipe
  with short rolling windows still reports enough history to build them.

## v0.3.0 (13/09/2026)

### Added

- `helpers.modeling` — linear, ridge and random forest regressors, a random
  forest classifier defaulting to balanced class weights, and metric
  functions for both regression and multiclass classification. Classification
  metrics report per-class recall and F1 for every label supplied, so a rare
  class that never appears in a split is still visible rather than silently
  omitted.
- `helpers.features` — `build_daily_features`, which assembles the complete
  feature set from raw hourly observations according to a declarative
  recipe, and `required_history_days`, which derives how much history a
  recipe needs to produce one row. Centralising this prevents
  training/serving skew: the recipe is stored in the model artefact, so a
  deployed service builds features exactly as the model was trained.
- `helpers.modeling.make_scaled_pipeline` — chains a scaler and an estimator
  into one object, so that scaling cannot be forgotten or applied
  differently at prediction time.
- `helpers.importance` — coefficient, impurity and permutation based feature
  importance, returned in a consistent tabular form.
- `helpers.visualization` — predicted against actual, residual, forecast
  timeline, feature importance and multiclass confusion matrix plots.

### Fixed

- `helpers.eda.numerical_summary` referenced an unimported `Literal` in its
  signature, and declared a permitted value that did not match its own
  default. The parameter is now annotated as `str`.
- Consolidated import statements in `helpers.visualization`, which had
  accumulated part-way down the module.

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

- `helpers.weather` — retrieval of historical observations from the
  Open-Meteo archive and forecast APIs, and aggregation of hourly
  observations into daily values.
- `helpers.targets` — the Climate Comfort Index, the Weather Hazard Index
  and the Weather Hazard Category, including their component scores.
- `helpers.timeseries` — lag, rolling window and cyclical calendar
  features, supervised frame construction, and chronological splitting with
  a horizon-length embargo at each boundary.
- Project hygiene: a `.gitignore` covering Python caches, build artefacts
  and virtual environments, and removal of previously committed
  `__pycache__` directories.

## v0.1.0 (20/08/2026)

- First release, as `36120_26sp_group36_25611094_package`.

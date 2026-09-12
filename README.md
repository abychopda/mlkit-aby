# mlkit-aby

A reusable Python machine learning toolkit that supports consistent,
reproducible experimentation across data science projects.

The package centralises general-purpose functionality so that experiment
notebooks are free to focus on experiment-specific decisions, analysis and
interpretation rather than repeating boilerplate.

## Design philosophy

> Reusable functionality belongs in the package; experiment-specific
> decisions remain in the notebooks.

## Installation

The package is published to [TestPyPI](https://test.pypi.org/):

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            mlkit-aby
```

The `--extra-index-url` is required because TestPyPI does not mirror the
package's dependencies (pandas, scikit-learn and so on), which are resolved
from the real PyPI.

## Package structure

```text
mlkit_aby/
│
└── helpers/
    ├── cleaning.py        # dataset cleaning utilities
    ├── eda.py             # exploratory data analysis summaries
    ├── features.py        # feature construction utilities
    ├── importance.py      # correlation and feature importance analysis
    ├── io.py              # dataset loading and saving
    ├── modeling.py        # model training and evaluation
    ├── preprocessing.py   # encoding and transformation
    ├── utils.py           # shared general-purpose utilities
    └── visualization.py   # plotting utilities
```

## Development

The project uses [uv](https://docs.astral.sh/uv/) for dependency management
and packaging.

```bash
# Install the project together with its development dependencies
uv sync

# Run the test suite
uv run pytest
```

## Publishing a new version

```bash
# 1. Bump the version in pyproject.toml and record the change in CHANGELOG.md
# 2. Build the distribution artefacts
uv build

# 3. Upload to TestPyPI
uv publish --index testpypi
```

## Author

**Abhishek Chopda** — Student ID **25611094**

Master of Data Science and Innovation
University of Technology Sydney (UTS)

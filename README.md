# Git2KG: Mining socio-technical relationships in GitHub repositories as temporal knowledge graphs

`Git2KG` is an OpenSource Python package, that mines data from repositories hosted on GitHub and extracts multi-relational temporal knowledge graphs from said data. It is capable of extracting predefined and custom features and labels for nodes and provides exports of temporal knowledge graph snapshots to three of the most common graph analysis package: `pathpyG` (https://github.com/pathpy/pathpyG), `torch-geometric` (https://github.com/pyg-team/pytorch_geometric) and `networkx` (https://github.com/networkx).

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) manages the environment. `uv.lock` is committed, so `uv sync` reproduces the dependency set the library is developed and tested against.

The mining dependencies are installed with the package. The three exporters are optional extras, so only the graph library you actually use is pulled in:

| Extra | Installs | Enables |
| --- | --- | --- |
| `pyg` | `torch`, `torch-geometric` | `PyGExporter` |
| `networkx` | `networkx` | `NetworkXExporter` |
| `pathpy` | `pathpyG` | `PathPyGExporter` |
| `all` | all of the above | every exporter |

`pathpyG` is not published on PyPI, so the `pathpy` extra installs it from a pinned git revision.

## Installation

From a clone:

```bash
uv sync --all-extras
```

This creates `.venv`, installs `git2kg` in editable mode and resolves everything from `uv.lock`. Use `--extra pyg` in place of `--all-extras` to install a single exporter's dependencies. Prefix commands with `uv run` to execute them in that environment, or activate `.venv` as usual.

To use the package outside the project, build a wheel and install it with the extras you need:

```bash
uv build
uv pip install "dist/git2kg-1.0.0-py3-none-any.whl[pyg,networkx]"
```

## Usage

1. Import the main class:

```python
from git2kg import Git2KG
from git2kg.export import PyGExporter
```

2. Mine a repository

```python
your_repo2kg = Git2KG.from_github_url("https://github.com/your/repository", github_token="token")

```

The token needs the scopes `public_repo` and `read:user` - or `repo` instead of `public_repo` for a private repository. `public_repo` alone mines everything except the email addresses of GitHub accounts, which are one of the signals that link an account to the author of a commit. Add `read:org` to identify actors that are organizations rather than people.

 Pass `github_api=REST` to mine over the REST API instead, which needs only `public_repo` but finds mentions with a regular expression and therefore might record code artifacts beginning with `@` in the issue or comment bodies as actors.

```python
from git2kg.dataloading import REST

your_repo2kg = Git2KG.from_github_url(url, github_token="token", github_api=REST)
```

3. Create snapshots

```python
your_repo2kg.freq = "ME" # monthly frequency
snapshots = your_repo2kg.get_snapshots()
exporter = PyGExporter()
hetero_data_dict = exporter.export_snapshots(snapshots)
```

Please check the notebooks in "demo" for a more detailed usage guide

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

`uv sync --all-extras` also installs the development tooling, which is declared as uv's default dependency group. Linting and type checking run through `pre-commit`; after cloning, install the hooks once so they run on every commit:

```bash
uv run pre-commit install
```

The hooks are `ruff check` and `mypy`, both of which can also be run directly:

```bash
uv run ruff check
uv run ruff format src
uv run mypy
```

`mypy` reads `[tool.mypy]` in `pyproject.toml`, which scopes it to `src`, so the hook, the command line and an editor all report the same thing.

### Running the tests

The suite is written with `unittest` and needs no further dependency:

```bash
uv run python -m unittest discover -s test -t .
```

`-t .` points discovery at the repository root, which the test modules need because they import each other's helpers absolutely, as `from test.test_gitdata import ...`. A full run takes roughly two minutes.

A single module or case can be run on its own:

```bash
uv run python -m unittest test.test_disambiguation
uv run python -m unittest test.test_disambiguation.TestDisambiguatedAuthorMapping
```


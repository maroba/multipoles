## How to contribute to *multipoles*

#### **Did you find a bug?**

* **Ensure the bug was not already reported** by searching on GitHub
  under [Issues](https://github.com/maroba/multipoles/issues).

* If you're unable to find an open issue addressing the
  problem, [open a new one](https://github.com/maroba/multipoles/issues/new). Be sure to include a **title and clear
  description**, as much relevant information as possible, and a **code sample** or an **executable test case**
  demonstrating the expected behavior that is not occurring.

#### **Did you write a patch that fixes a bug?**

* Open a new GitHub pull request with the patch.

* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

#### **Do you intend to add a new feature or change an existing one?**

* Suggest your change in the [multipoles discussion forum](https://github.com/maroba/multipoles/discussions) and
  start writing code.

* Do not open an issue on GitHub until you have collected positive feedback about the change. GitHub issues are
  primarily intended for bug reports and fixes.

#### **Setting up a development environment**

Clone the repository and install *multipoles* in editable mode, together with the tools for testing and
the documentation:

```
git clone https://github.com/maroba/multipoles.git
cd multipoles
python -m pip install -e . pytest ruff -r docs/requirements.txt
```

Before you open a pull request, please make sure that

* **the tests pass**: `pytest`. This also runs the examples in the docstrings, the README and the usage
  guide, so if you change an example, check that it still runs.
* **the code is clean**: `ruff check .`
* **the documentation builds without warnings**: `mkdocs build --strict`. This also runs the notebooks in
  `docs/`. Use `mkdocs serve` to preview the documentation at http://127.0.0.1:8000.
* **new features or behavior changes have tests** and are described in the docstrings, in the documentation
  in `docs/`, and in [CHANGELOG.md](CHANGELOG.md) under *Unreleased*.

#### **Do you have questions about the source code?**

* Ask any question about how to use *multipoles* in
  the [discussion forum](https://github.com/maroba/multipoles/discussions).

Thanks! :heart: :heart: :heart:

The *multipoles* Team

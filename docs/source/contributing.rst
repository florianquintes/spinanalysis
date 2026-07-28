Contributing
============

Thank you for contributing to ``spinanalysis``. The project is a Python
framework for loading, processing, plotting, simulating, optimizing, and
saving electron paramagnetic resonance (EPR) spectra.

Project Structure
-----------------

The main package modules are organized by workflow:

* ``core.py`` orchestrates simulation and optimization.
* ``epr.py`` defines EPR parameter and workflow objects.
* ``loading.py`` reads EPR data files.
* ``processing.py`` transforms and reconstructs spectra.
* ``plotting.py`` renders spectra and figures.
* ``saving.py`` writes figures, simulated data, and output files.
* ``profiles.py`` manages configuration and profile files.
* ``_wrappers.py`` provides internal timing and multiprocessing wrappers.
* ``_interface_handler.py`` provides internal simulation and optimization
  interfaces.

Profile templates and plotting styles are package data under
``src/spinanalysis/data/profiles``. Changes to a profile schema should update
the corresponding ``configspec.ini`` and relevant fixtures under
``tests/data``.

Development Setup
-----------------

The project requires Python 3.13 or newer. The repository uses ``uv`` for
dependency and environment management.

Clone the repository and change into its directory:

.. code-block:: console

   git clone https://github.com/florianquintes/spinanalysis.git
   cd spinanalysis

Install the locked development environment from the repository root:

.. code-block:: console

   uv sync --dev

The package uses a ``src`` layout. Application code belongs under
``src/spinanalysis`` and tests belong under ``tests``.

Branches
--------

Create a focused branch for each change and keep unrelated changes separate.
The repository does not currently document a required branch naming scheme.

Pull Requests
-------------

Pull requests should explain the change and include the relevant validation
results. Before opening a pull request, run the checks that apply to the
change:

.. code-block:: console

   uv run pytest
   uv run ruff check .
   uv build

If documentation is changed, also build the documentation as described in the
Documentation section. Mention known baseline failures or warnings rather
than presenting them as regressions.

Code Style
----------

Follow the existing Python and reStructuredText style in the surrounding
files. Keep importable Python modules under ``src/spinanalysis`` and add tests
under ``tests``. The repository has no configured formatter or type checker.

Ruff is the configured code-quality tool, using its default rules:

.. code-block:: console

   uv run ruff check .

Documentation
-------------

Documentation source files are under ``docs/source``. The Sphinx configuration
adds ``src`` to the module search path and reads the package version from
``pyproject.toml``.

Build the documentation from the ``docs`` directory:

.. code-block:: console

   uv run sphinx-build -M html source build

The generated HTML is written to ``docs/build/html``. Add a title to every
document included by the root ``index.rst`` so Sphinx can include it in the
table of contents.

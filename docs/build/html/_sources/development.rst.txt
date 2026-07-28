Development
===========

Development Setup
-----------------

The project requires Python 3.13 or newer and uses ``uv`` for dependency and
environment management. From the repository root, install the development
environment with:

.. code-block:: console

   uv sync --dev

Running Tests
-------------

Run the test suite from the repository root:

.. code-block:: console

   uv run pytest

Code Formatting and Linting
---------------------------

Check the code with Ruff:

.. code-block:: console

   uv run ruff check .

The repository does not currently configure a separate formatter.

Build the Documentation
-----------------------

Build the HTML documentation from the ``docs`` directory:

.. code-block:: console

   uv run sphinx-build -M html source build

The generated documentation is written to ``docs/build/html``.

Getting Started
===============

This guide will walk you through setting up the ``telemetry-anomdet`` package for development and use.

Installation
------------

Contributors use `uv <https://docs.astral.sh/uv/>`_ to manage the project environment.
uv handles virtual environment creation and dependency installation automatically -- no
manual ``venv`` setup required.

**1. Install uv:**

.. code-block:: bash

    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows (PowerShell)
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

**2. Clone the repository and install:**

.. code-block:: bash

    git clone https://github.com/JulioAnzaldo/telemetry-anomdet.git
    cd telemetry-anomdet
    uv sync

``uv sync`` creates ``.venv`` automatically, installs all core dependencies, and
generates ``uv.lock`` for reproducible installs. No separate activate step is needed
-- uv commands run inside the environment automatically.

To include optional dependencies:

.. code-block:: bash

    uv sync --extra ccsds      # CCSDS packet support
    uv sync --group dev        # testing and linting tools
    uv sync --group docs       # Sphinx and documentation tools
    uv sync --all-groups       # everything (recommended for contributors)

Running Tests
-------------

.. code-block:: bash

    uv sync --group dev
    uv run pytest -v

Basic Usage
-----------

You can now import and use the library. For example, to check the installed version:

.. code-block:: python

    import telemetry_anomdet
    print(telemetry_anomdet.__version__)

Contributing
------------

The ``telemetry-anomdet`` project uses a feature-branch workflow. All new work should be done on a dedicated branch created from the main development branch.

**1. Switch to the development branch:**
Ensure your local repository is up-to-date with the remote ``dev`` branch.

.. code-block:: bash

    git switch dev
    git pull origin dev

**2. Create a new feature branch:**
Create a new branch for your specific feature or bug fix. This keeps your changes isolated.

.. code-block:: bash

    git switch -c feature/new-feature

**3. Sync the environment:**

.. code-block:: bash

    uv sync --all-groups       # installs core + dev + docs dependencies
    uv sync --extra ccsds      # add CCSDS support if needed

**4. Push your branch and open a pull request:**
Push your branch to the remote repository and open a pull request against the ``dev`` branch.

.. code-block:: bash

    git push -u origin feature/new-feature

This ensures your code is reviewed and tested before being merged into the main development branch.
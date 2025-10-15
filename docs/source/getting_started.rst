Getting Started
===============

This guide will walk you through setting up the ``telemetry-anomdet`` package for development and use.

Installation
------------

It is highly recommended to use a Python virtual environment to manage project dependencies. This ensures that the packages you install for this project do not interfere with other projects on your system.

**1. Clone the repository and navigate to the project directory:**

.. code-block:: bash

    git clone https://github.com/JulioAnzaldo/telemetry-anomdet.git
    cd telemetry-anomdet

**2. Create and activate a virtual environment:**

.. code-block:: bash

    # On macOS or Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    py -m venv venv
    venv\Scripts\activate

Your command line prompt should now show ``(venv)`` to indicate the virtual environment is active.

**3. Install the package locally in editable mode:**

The package can be installed with different sets of dependencies. The core library supports CSV files by default. For CCSDS support, install the optional ``ccsds`` dependency.

.. code-block:: bash

    # Install the core library with CSV support
    pip install -e .

    # Install with optional CCSDS support (recommended for full functionality)
    pip install -e .[ccsds]

Running Tests
-------------

To verify that the package and its dependencies are installed correctly, you can run the test suite. This requires the ``dev`` dependencies to be installed, which include ``pytest``.

.. code-block:: bash

    # First, install development dependencies (if not already done)
    pip install -e .[dev]

    # Run the tests
    pytest -v

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

    git switch -c feature/my-new-feature

**3. Install the package locally in editable mode:**

The package can be installed with different sets of dependencies. The core library supports CSV files by default. Optional dependencies are available for development, documentation, and CCSDS support.

.. code-block:: bash

    # Install the core library with CSV support
    pip install -e .

    # Install with optional CCSDS support
    pip install -e .[ccsds]

    # Install development dependencies (tests, formatting, linting)
    pip install -e .[dev]

    # Install documentation dependencies (Sphinx, themes, etc.)
    pip install -e .[docs]

    # Install everything (recommended for contributors)
    pip install -e .[dev-all]

**4. Push your branch and open a pull request:**
Push your branch to the remote repository and open a pull request against the ``dev`` branch.

.. code-block:: bash

    git push -u origin feature/my-new-feature

This ensures your code is reviewed and tested before being merged into the main development branch.
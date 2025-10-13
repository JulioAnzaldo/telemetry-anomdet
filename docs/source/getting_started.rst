Getting Started
===============

Installation
------------

Clone the repository and install the package locally:

.. code-block:: bash

    git clone https://github.com/JulioAnzaldo/telemetry-anomdet.git
    cd docs/source
    python -m pip install -e ./src

Running Tests
-------------

To verify the package is installed correctly:

.. code-block:: bash

    pytest -v

Basic Usage
-----------

.. code-block:: python

    import telemetry_anomdet
    print(telemetry_anomdet.__version__)

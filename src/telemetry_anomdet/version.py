# telemetry_anomdet/version.py

"""
Package version.
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError # type: ignore[assignment]

try:
    __version__ = version("telemetry-anomdet")
except PackageNotFoundError:
    __version__ = "0.0.0"

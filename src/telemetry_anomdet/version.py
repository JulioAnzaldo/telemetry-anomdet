# telemetry_anomdet/version.py

"""
Package version.
"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # Python < 3.8
    from importlib_metadata import PackageNotFoundError, version  # type: ignore[assignment]

try:
    __version__ = version("telemetry-anomdet")
except PackageNotFoundError:
    __version__ = "0.0.0"

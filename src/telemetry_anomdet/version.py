# telemetry_anomdet/version.py

"""
Package version.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("telemetry-anomdet")
except PackageNotFoundError:
    __version__ = "0.0.0"

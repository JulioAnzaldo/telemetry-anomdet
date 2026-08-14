# ESP-IDF project for the distilled detector conformance harness.
#
# Generate the sources into main/generated/ before building:
#   python examples/export_kangdn_c.py targets/esp32s3/main/generated
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(kangdn_conformance)

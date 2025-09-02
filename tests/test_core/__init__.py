"""Core tests for the :mod:`m3c2` package.

This suite exercises the fundamental building blocks used by the
M3C2 pipeline.

Overview
--------
The tests contained here validate:

* Statistical utilities such as basic metric computation and outlier
  detection.
* Bounding-box helpers that read point clouds, transform coordinates
  and clip to oriented boxes.
* Point-cloud quality assessments computing density, height ranges and
  nearest-neighbour distances.
* Export helpers that write statistical tables to Excel or JSON.
* The M3C2 orchestration runner that invokes the underlying Py4DGeo
  implementation with the correct parameters.
* Scale parameter estimation routines used to select normal and
  projection scales.

Together these tests provide confidence that the low-level components
supporting the higher-level M3C2 workflow behave as expected.
"""

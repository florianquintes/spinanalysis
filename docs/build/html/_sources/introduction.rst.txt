Introduction
============

``spinanalysis`` is a Python framework for the analysis of electron
paramagnetic resonance (EPR) spectra. It is designed to support a complete
workflow, from loading experimental data and applying common processing steps
to visualizing, simulating, optimizing, and saving results.

The package supports common EPR data formats, including Bruker BES3T,
ESP380E, MATLAB, simulated, and plain-text data. Processing tools cover tasks
such as normalization, offset reduction, background correction, and
reconstruction of out-of-phase ESEEM spectra. Matplotlib-based plotting
functions provide consistent two-dimensional and three-dimensional figures
using configurable plot styles.

Simulation and optimization workflows are organized around EPR parameter
objects and profile-based configuration. The profile system stores reusable
settings for spin systems, simulation and optimization routines, plotting,
and saving. Some specialized simulation routines are provided through
external dependencies and may require additional installation steps.

Module overview
---------------

The package is organized into modules that support different parts of an EPR
analysis workflow:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Description
   * - ``core``
     - Orchestrates simulation and optimization workflows.
   * - ``epr``
     - Defines EPR parameter, experimental, simulation, and optimization
       objects.
   * - ``loading``
     - Loads EPR data from Bruker, ESP380E, MATLAB, simulated, and text files.
   * - ``processing``
     - Normalizes, corrects, and reconstructs spectra.
   * - ``plotting``
     - Creates two-dimensional and three-dimensional spectrum plots.
   * - ``saving``
     - Saves figures, simulated data, and simulation output files.
   * - ``profiles``
     - Creates, loads, saves, imports, and exports configuration profiles.
   * - ``_wrappers``
     - Provides internal timing and multiprocessing wrappers.
   * - ``_interface_handler``
     - Provides internal interfaces for simulation and optimization routines.

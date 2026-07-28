# spinanalysis

`spinanalysis` is a high-level Python framework for the analysis of electron
paramagnetic resonance (EPR) spectra.

The framework provides tools for loading, processing, plotting, simulating,
optimizing, and saving EPR spectra. It is designed to support complete analysis
workflows from experimental data to simulated and fitted spectra.

## Features

- High-level workflow for EPR spectral analysis
- Loading of:
  - Bruker `.bes3t` files
  - Bruker ESP380E files
  - `.txt` files
- Spectral processing:
  - Normalization
  - Offset reduction
  - Background correction
  - Reconstruction of out-of-phase (OOP) ESEEM spectra
- Two-dimensional and three-dimensional plotting based on Matplotlib
- Consistent plot designs using custom stylesheets
- Profile-based configuration for:
  - Spin systems
  - Simulation routines
  - Optimization routines
  - Optimization parameter ranges
  - Plot settings
  - Saving settings
- Simulation and optimization of EPR spectra
- EasySpin-inspired classes for simulation and optimization workflows:
  - `Sys`
  - `Exp`
  - `SimOpt`
  - `FitOpt`
  - `Var`
- Simulation of spin-correlated, singlet-born radical pairs
- Simulation of out-of-phase (OOP) ESEEM signals
- Interface to `teacups`
- Saving processed and simulated spectra as `.txt` files

## Requirements

- Python 3.13 or newer

Additional dependencies for the simulation routines are currently under
development and will be published separately.

## Installation

### Using uv

```bash
uv add spinanalysis
```

### Using pip

```bash
pip install spinanalysis
```

## Development

Clone the repository and install the development dependencies with `uv`:

```bash
git clone https://github.com/florianquintes/spinanalysis.git
cd spinanalysis
uv sync --dev
```

Run the tests:

```bash
uv run pytest
```

## Documentation

The documentation is available on
[GitHub Pages](https://florianquintes.github.io/spinanalysis/).

## License

This project is licensed under the GNU General Public License v3.0.
See the [LICENSE](LICENSE) file for details.
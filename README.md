# Elastic2D
A few simple pre- and post-processing scripts to facilitate two-dimensional elastic constants calculations via the stress-strain method.

- `elastic2D_tools.py` contains the main script that only requires `Numpy` to generate the strained lattices and post-process the outputs.
- `vasp_elastic2D.py` is an example of how the contents of `elastic2D_tools.py` can be used to compute and post-process elastic constants with `VASP`. 
This script makes use of `Pymatgen` for setting up the `VASP` calculations.

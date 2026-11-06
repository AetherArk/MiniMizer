# MiniMizer

A simple minimization pipeline for cleaning up molecular structures from the PDB, cofolding output, or docking.

This method is designed to be simple, fast, and reproducible.  

## Workflow (v 0.1)
- MMFF94s ligand repair
- OpenFF/CHARMM36 pocket minimization.

**This will be modified to a fully OpenFF workflow soon**

## Dependencies

- Stage 1: `requirements.txt`
- Stage 2: `requirements-md.txt`
- Trusted input packets only; do not use Python `-O`.

## Run

```sh
python minimize.py example/config.json replay --md-python /path/to/md/python
```

- Input schema `2.0`: one JSON file; every file path resolves from its parent, not the working directory. See `example/config.json` and `schemas/config.schema.json`.
- `ligand`: starting complex PDB (angstrom), SMILES, atom mapping, state and fixed indices.
- `receptor`: prepared coordinates (nm), ordered atom metadata, heme names and frozen state; `forcefield`: merged CHARMM36/OpenFF OpenMM XML (masses, bonded terms, charges, nonbonded parameters/exclusions and constraints). No reparameterization.
- `protocol: "0.1"`: fixed MMFF94s → pocket preset, including optimizer constants, stereo/selection gates, rounding and mobility (chain A sidechains within 5 Å, fixed backbone/Cys442 SG/heme and ligand anchors); heavy-atom restraint 12552 kJ mol⁻¹ nm⁻² with fresh references. These are not adjustable knobs. Output schema remains `1.0`.

## Tests

```sh
python -m unittest discover -s tests -v
```

## Example
- CYP3A4 protein-ligand cofolding

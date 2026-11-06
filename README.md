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
python minimize.py example replay --md-python /path/to/md/python
```

## Tests

```sh
python -m unittest discover -s tests -v
```

## Example
- CYP3A4 protein-ligand cofolding

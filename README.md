# MiniMizer 0.1

- Historical CYP minimization from saved prepared-state inputs.
- MMFF94s ligand repair → OpenFF/CHARMM36 pocket minimization.
- Frozen protein states; no raw-structure preparation.

## Dependencies

- Stage 1: `requirements.txt`
- Stage 2: `requirements-md.txt`
- Trusted input packets only; do not use Python `-O`.

## Run

```sh
python minimize.py example replay --md-python /path/to/md/python
```

- Alternative: `--md-image <existing-image>`; no image download.
- Output directory must be new.
- Outputs: coordinates, candidate PDB, optimization and verification reports.

## Tests

```sh
python -m unittest discover -s tests -v
```

- Included example: exact historical coordinate reproduction.
- PoseBusters not rerun.

## License

- Apache-2.0; code adapted from ADMET-CYP.

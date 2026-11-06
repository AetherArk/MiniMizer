# MiniMizer 0.1

- MMFF94s ligand repair → OpenFF/CHARMM36 pocket minimization.
- Prepared force fields and receptor states; no raw-structure preparation.

## Run

```sh
python minimize.py example output --md-image local/md_openmm:0.1
# or: --md-python /path/to/md/python
```

- Dependencies: `requirements.txt`; MD runtime: `requirements-md.txt`.
- Existing Docker image only; offline, one CPU, 4 GB. Output directory must be new.
- Trusted prepared inputs only; do not use Python `-O`.

## Inputs

- `input.pdb`: starting ligand coordinates and receptor heavy atoms; explicit ligand CONECT graph.
- JSON interface `1.0` (separate from protocol `0.1`): [`config` schema](schemas/config.schema.json), [`report` schema](schemas/report.schema.json); enforced before optimization / success; integer fields require integer JSON tokens.
- `config.json`: authoritative ligand SMILES, atom mapping, chemical state, fixed anchors, heme-name mapping.
- `system.xml.gz`: prepared OpenFF/CHARMM parameters in receptor-then-ligand atom order.
- `atoms.json.gz`: receptor particle identities/order, including hydrogens.
- `receptor-positions.npy`: prepared receptor coordinates, nm, in that order.
- New pose: copy `example`, replace ligand coordinates in `input.pdb`; preserve graph/identities, receptor and chemistry. Restraints and pocket selection use the actual start.

## Outputs / checks

- `prepared_only.pdb`: MMFF-selected pose.
- `local/k12552/candidate.pdb`, `positions.npy`: minimized candidate; retained even if validation fails.
- `report.json`: actual convergence, fixed-atom and boundary-check status; nonzero exit on failure. Execution failures carry `error`, without fabricated metrics; invalid inputs stop before output creation.
- Boundary energy tolerance: `1e-5` kJ/mol; unchanged v0.1 settings. PoseBusters not run.

## Tests

```sh
python -B -m unittest discover -s tests -v
MINIMIZER_MD_IMAGE=local/md_openmm:0.1 python -B -m unittest discover -s tests -v
```

- Optional offline integration: +0.01 Å on one ligand coordinate; functional test, not a new experimental dataset or known-answer replay.

## License

- Apache-2.0; code adapted from ADMET-CYP.

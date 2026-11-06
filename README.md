# MiniMizer — protocol 0.1

Replay the historical two-stage CYP minimization from an accepted prepared-state
packet. This is **not a raw-structure preparation tool**. It does not run PROPKA,
assign new protein states, generate cofolds, or use an OpenFF-only receptor.

## Run

Use two Python environments to reproduce the historical runtime:

- Stage 1: RDKit 2026.03.2, NumPy 2.1.3, SciPy 1.17.1.
- Stage 2: OpenMM 8.3.1, RDKit 2025.03.6, NumPy 1.26.4, SciPy 1.15.2.

No OpenFF toolkit, PDBFixer, PoseBusters or parameter download is needed.
The packet contains frozen parameters. Load only trusted packets and RDKit files.

```sh
python minimize.py example replay --md-python /path/to/md/python
python -m unittest discover -s tests -v
```

Alternatively, use an **existing** Docker image with the Stage 2 dependencies:

```sh
python minimize.py example replay --md-image local/md_openmm:0.1
```

The image name is a local runtime option, not a published dependency. The command
does not pull an image. Docker runs offline with one CPU and 4 GB RAM. Choose a
new output directory for each run. Do not use Python `-O`.

## Method

1. Run the original ligand-only MMFF94s stereo-repair arms. Position constants
   are 10, 30, 100 and 300 kcal/mol/Å²; signed-volume constants are 100 and 1000
   kcal/mol/Å⁶. Each arm uses L-BFGS-B, at most 3000 iterations, `gtol=1e-4`,
   `ftol=1e-13`, `maxls=50`, `maxcor=20`. Source axial anchors stay fixed.
   Select the lowest-RMSD converged graph/stereo/tetrahedron-valid arm within
   0.5 Å RMSD and 1 Å maximum movement. Break ties by arm name. No PB prefilter.
2. Use the accepted CHARMM36 protein/heme and OpenFF unconstrained 2.2.1 ligand
   parameters, including their historical charges and exceptions. Select whole
   protein residues within 5 Å of the Stage 1 ligand. Move their side chains
   and the ligand, except fixed axial atoms. Backbone, heme, Cys442 SG and the
   remaining protein stay fixed. Restrain movable heavy atoms at
   12552 kJ/mol/nm². Retain original bonded boundary terms and local nonbonded
   terms, including CHARMM NBFIX. Exclude distant protein nonbonded terms.
   Run Reference-platform L-BFGS-B with at most 500 iterations, `gtol=10`,
   `ftol=1e-12`, `maxls=30`, `maxcor=10`. This is minimization, not dynamics.

The frozen receptor retains the historical PROPKA-derived assignments and any
accepted corrections. It does not substitute the later no-PROPKA policy.
Stage 1 PDB rounding and the historical float32 RDKit handoff are preserved.

## Inputs and outputs

`example/manifest.json` binds the input hashes. The packet contains the
pre-Stage-1 PDB, ligand graph mapping, fixed axial selection, prepared receptor
coordinates, atom identities, assigned states and compressed full-system XML.
`expected_start.npy` binds the accepted restraint reference; Stage 1 must
reconstruct it exactly. `expected_final.npy` is a comparison target, never an
optimizer starting point. Other packets require these same accepted historical
artifacts. Raw input preparation and generic engine adapters are outside 0.1.

The output contains every Stage 1 arm report, the selected PDB, the exact Stage 2
coordinate array, reduced system, index map, candidate PDB and `report.json`.
The candidate PDB retains source atom names and residue numbers. Exact arrays
are authoritative; PDB coordinates are rounded to 0.001 Å. The report gives
actual iterations, residual gradients, fixed-atom movement and reproduction
comparisons. The original boundary threshold remains **1e-5 kJ/mol**. A failed
boundary check is reported as a failure, not a reason to loosen the threshold.
Reproduction failure returns a nonzero exit code and retains diagnostic output.

## Verified example and limits

`alphafold3/OCNT-2312554/seed-11/sample-3` was selected through the historical
public-coordinate feature list. Its archived status is `REFINED_PASS`.
Both stages were executed: all eight Stage 1 arms matched historical iteration,
energy, gradient and movement values. The chosen arm was
`k300_ks1000_orientation0`. Stage 2 took 106 iterations and 115 evaluations.
The final array and candidate PDB were **byte-identical** to the archived files.
See [example/REPRODUCTION.json](example/REPRODUCTION.json) for measured results.

This verifies one anchored AF3-derived example, not every engine, ligand or
stereochemical case. No physical/PoseBusters checks are rerun. Historical PASS
is provenance, not a new retention decision. All output is labeled
`REPLAY_DIAGNOSTICS_ONLY`. Different runtimes can change optimizer trajectories;
byte identity is established only for the versions above. The historical
cross-version RDKit handoff emits a warning; this example passed exact checks.

Original code: ADMET-CYP, Apache-2.0; see [LICENSE](LICENSE) and
[SOURCE.json](SOURCE.json). Numerical kernels are extracted from the original
historical route, not the newer processing package. The license file covers
source-code attribution; it does not relicense third-party force fields or
model software. No model weights are included.

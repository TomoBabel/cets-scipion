# cets-scipion

Converters from [Scipion](https://scipion.i2pc.es/) projects to CETS, the cryo-electron
tomography metadata standard defined in
[cets-data-models](https://github.com/TomoBabel/cets-data-models). They read the sqlite files
in which Scipion stores its sets of objects (`Runs/<protocol>/*.sqlite`).

| Scipion set | CETS objects |
|---|---|
| `SetOfTiltSeries` | `TiltSeries`, `Alignment`, `Instrument`, `AcquisitionSession` |
| `SetOfCTFTomoSeries` | `CTFMetadata` (one per tilt-image) |
| `SetOfTomograms` | `Tomogram` |
| `SetOfCoordinates3D` | `PointSet3D`, or `PointMatrixSet3D` if the coordinates are oriented (one per tomogram) |
| `SetOfSubTomograms` | `PointSet3D`/`PointMatrixSet3D` (picked coordinates) and `Average` with one `ParticleMap` per subtomogram |

Only the Scipion → CETS direction exists for now.

## Installation

Python 3.11 or newer. The CETS data model is not declared as a dependency, so install it first:

```bash
pip install -e /path/to/cets-data-models
pip install -e .          # or: pip install -e .[dev] for the development tools
```

This installs the `scipion` package and the `cets-scipion` command. Scipion itself is not
needed: the sqlite files are read directly.

## Command line

```text
cets-scipion <data type> to-cets [options]
```

Each command writes a CETS yaml file with one entry per kind of object (`tilt_series`,
`alignment`, `instrument`, `acquisition_session`, `ctf`, `tomogram`, `coordinates`, `average`);
the values are lists, as a Scipion set may hold several tilt-series or tomograms. Every
sub-command has `--help`. Errors print a one-line message and exit with code 1; put `--debug`
before the data type to also print the traceback. Input files are only read.

```bash
P=ScipionUserData/projects/my_project/Runs

# Tilt-series, with their CTF and the nominal tilt angles of the imported set
cets-scipion tilt-series to-cets \
    --ts-db $P/000128_ProtImodImportTransformationMatrix/tiltseries.sqlite \
    --nominal-ts-db $P/000002_ProtImportTs/TiltSeries.sqlite \
    --ctf-db $P/000084_ProtImportTsCTF/ctftomoseries.sqlite -o ts.yaml

# CTF of one tilt-series (--ts-id is required if the set has several)
cets-scipion ctf to-cets --ctf-db $P/000084_ProtImportTsCTF/ctftomoseries.sqlite --ts-id TS_03 -o ctf.yaml

# Tomograms
cets-scipion tomogram to-cets --tomo-db $P/000198_ProtImportTomograms/tomograms.sqlite -o tomos.yaml

# Coordinates (all the tomograms, or the ones given with --tomo-id, which can be repeated)
cets-scipion coordinates to-cets --coords-db $P/000275_ProtImportCoordinates3DFromStar/coordinates3d.sqlite \
    --tomo-db $P/000198_ProtImportTomograms/tomograms.sqlite --pixel-size 5.4 -o coords.yaml

# Subtomograms
cets-scipion subtomograms to-cets --subtomo-db $P/000943_EmanProtTomoExtraction/subtomograms.sqlite \
    --tomo-db $P/000198_ProtImportTomograms/tomograms.sqlite --tomo-id TS_03 -o subtomos.yaml
```

| Option | Meaning |
|---|---|
| `--nominal-ts-db` | The set of tilt-series before the alignment, to read the nominal tilt angles from (see below). |
| `--ctf-db` | CTF set whose CTFs are added to the tilt-images. |
| `--tomo-db` | Tomograms the coordinates/particles were picked in: their origin refers the coordinates to the corner of the tomogram. For subtomograms it is optional (default: the centre of the tomogram of each subtomogram, Scipion's default origin). |
| `--tomo-id` | Tomogram(s) to convert (Scipion `tsId`). Default: all. |
| `--pixel-size`, `--coords-pixel-size` | Voxel size (Å) of the tomograms the coordinates refer to. With it, the coordinates also carry the physical coordinate system and its `array_to_physical` scale. |

## Python API

```python
from scipion.converters.ctf_set import ScipionSetOfCtf
from scipion.converters.tilt_series_set import ScipionSetOfTiltSeries
from scipion.converters.coodinates3d import ScipionSetOfCoordinates3D

ctfs = ScipionSetOfCtf("Runs/000084_ProtImportTsCTF/ctftomoseries.sqlite").scipion_to_cets()
tilt_series, alignments, instruments, sessions = ScipionSetOfTiltSeries(
    "Runs/000128_ProtImodImportTransformationMatrix/tiltseries.sqlite"
).scipion_to_cets(ctf_md=ctfs, nominal_ts_db_path="Runs/000002_ProtImportTs/TiltSeries.sqlite")
points = ScipionSetOfCoordinates3D("Runs/.../coordinates3d.sqlite").scipion_to_cets(
    "TS_03", tomo_db_path="Runs/000198_ProtImportTomograms/tomograms.sqlite"
)
```

The other converters are `ScipionSetOfTomograms` (`converters/tomograms_set.py`) and
`ScipionSetOfSubtomogras` (`converters/subtomograms.py`). The sqlite path must be inside a
Scipion project (`<project>/Runs/<protocol>/`), as the file names it stores are relative to
the project.

## Conventions

- **Pixel sizes** are Scipion's sampling rates (`_samplingRate`) or, if a set does not store
  them, those of the file headers. If neither is known, an error is raised.
- **Tilt angles.** Scipion keeps one angle per tilt-image (`_tiltAngle`), which the alignment
  protocols overwrite with the refined one. For aligned tilt-images (they have a
  transformation matrix) it is stored as `ProjectionAlignment.refined_tilt_angle`, and the
  nominal angle is read from `--nominal-ts-db` or left empty. For tilt-images not aligned, it
  is `TiltImage.nominal_tilt_angle`.
- **Alignment.** The transformation matrix of each tilt-image becomes a `ProjectionAlignment`
  with `[Affine, Translation]` (rotation, then the 2-D shift in Å). Tilt-series that were not
  aligned get the identity.
- **CTF** is matched with its tilt-image as Scipion does: by acquisition order or, if it is not
  stored, by index. Defoci are in Å; phase shifts in degrees.
- **Acquisition.** The voltage, spherical aberration, amplitude contrast and tilt axis angle are
  read from the acquisition of each tilt-series (`TomoAcquisition`, in its row of the set).
  Tilt-series with the same values share an `AcquisitionSession`, and sessions with the same
  voltage an `Instrument`.
- **Coordinates** are referred to the corner of their tomogram (0-based voxel positions,
  Scipion's `BOTTOM_LEFT_CORNER`), not to its centre as Scipion stores them.
- **Orientations.** Scipion stores the transformation from the particle to the reference; CETS
  stores the opposite (reference/template → particle, `p = R @ r + t`). So the coordinate
  orientations (`PointMatrixSet3D.matrix3D`) and the subtomogram poses (`Affine` +
  `Translation` of each `ParticleMap`, shifts in Å) are the inverse of the Scipion matrices.
  This makes them identical to the ones of cets-dynamo, cets-gapstop and cets-pytom_tm for the
  same particles. Coordinates whose matrices are all the identity have no orientation and are
  a `PointSet3D`.
- **Sections** are 0-based (Scipion's `_index` is 1-based).

## Limitations

- No CETS → Scipion direction yet.
- Disabled tilt-images (`_objEnabled`) are converted like the rest: CETS has no field to mark
  excluded views yet.
- The rotation centre of the tilt-image alignment is not stored (Scipion, like IMOD, rotates
  about the centre of the image, while the CETS tilt-image frame has its origin at the
  corner); this waits for a decision on the frame of `ProjectionAlignment`.
- RELION 5 pseudo-subtomograms (2-D stacks) are read as volumes; they are left to the
  cets-relion converters.
- Not converted yet: masks (`SetOfTomoMasks`), meshes (`SetOfMeshes`), score tomograms,
  classes and averages of subtomograms, landmark models and tilt-series movies.

## Development

```bash
pip install -e .[dev]
pre-commit install
pytest scipion/tests/tests_*.py   # the test files must be given: they are named tests_*.py
ruff check . && ruff format .
mypy scipion
```

### Tests

| File | What it tests |
|---|---|
| `scipion/tests/tests_cets_scipion.py` | The coordinates converter, on small synthetic sets. |
| `scipion/tests/tests_cli.py` | Every `cets-scipion` sub-command, on small synthetic Scipion projects (built in a temporary directory). |
| `scipion/tests/tests_cli_fixtures.py` | Every `cets-scipion` sub-command, on the sets of a real Scipion project (`scipion/tests/data`), checking the CETS yaml against the values stored in the sqlite files. |

```bash
pytest scipion/tests/tests_cli_fixtures.py                          # one file
pytest scipion/tests/tests_cli_fixtures.py -k coordinates           # some tests
```

The test data (`scipion/tests/data`) holds two tilt-series, `TS_03` (40 tilt-images) and
`TS_54` (41), processed in Scipion:

| File | Content |
|---|---|
| `imported_tilt_series.sqlite` | `SetOfTiltSeries` as imported (nominal tilt angles). |
| `aligned_tilt_series.sqlite` | The same tilt-series, aligned (refined tilt angles and transformation matrices). |
| `ctftomoseries.sqlite` | `SetOfCTFTomoSeries` of both tilt-series. |
| `tomograms_bin8_no_fid.sqlite` | `SetOfTomograms` of both tilt-series, binned 8 times. |
| `coordinates3d.sqlite` | `SetOfCoordinates3D`: 95 oriented coordinates picked in the tomogram of `TS_03`. |
| `subtomograms.sqlite` | `SetOfSubTomograms` extracted from those coordinates. |
| `ts_ids.txt` | The `tsId` of the tilt-series, one per line (`TS_03`, `TS_54`). |
| `coordinates_pixel_size.txt` | Voxel size of the tomograms the coordinates refer to (10.8 Å/voxel). |
| `unbinned_pixel_size.txt` | Pixel size of the unbinned tilt-series (1.35 Å/pixel). |

The tests pass the values of the text files to the command line (`--ts-id`, `--tomo-id`,
`--pixel-size`, `--coords-pixel-size`) and compare the outputs with them. The images are not
part of the test data: the tests copy the sqlite files to a temporary Scipion project
(`<project>/Runs/<protocol>/`) and create there, at the paths stored in the sets, empty MRC
files with the dimensions of the real ones (sparse files, so they take no disk space), as the
converters only read their headers. The files in `scipion/tests/data` are never modified.

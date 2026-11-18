# -*- coding: utf-8 -*-
"""Tests of the ``cets-scipion`` command line with the data of a real Scipion project
(``scipion/tests/data``): the sets of two tilt-series (TS_03 and TS_54), their CTFs, their
tomograms (binned 8 times) and the coordinates / subtomograms picked in TS_03.

Fixtures in ``scipion/tests/data``:

- ``imported_tilt_series.sqlite``: SetOfTiltSeries as imported (nominal tilt angles).
- ``aligned_tilt_series.sqlite``: the same tilt-series with their alignment (refined
  tilt angles and transformation matrices).
- ``ctftomoseries.sqlite``: SetOfCTFTomoSeries of both tilt-series.
- ``tomograms_bin8_no_fid.sqlite``: SetOfTomograms of both tilt-series.
- ``coordinates3d.sqlite``: SetOfCoordinates3D picked in the tomogram of TS_03.
- ``subtomograms.sqlite``: SetOfSubTomograms extracted from those coordinates.
- ``ts_ids.txt``: the tsId of the tilt-series of the dataset, one per line.
- ``coordinates_pixel_size.txt``: voxel size (Å) of the tomograms the coordinates refer to.
- ``unbinned_pixel_size.txt``: pixel size (Å) of the (unbinned) tilt-series.

The fixtures are copied to a temporary Scipion project (``<project>/Runs/<protocol>/``), as
the file names stored in the sets are relative to the project. The images are not part of
the fixtures: the ones the sets refer to are created there as empty (sparse) MRC files with
the dimensions of the real ones, as only their headers are read.
"""

import ast
import contextlib
import io
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import TestCase

import mrcfile
import numpy as np

from cets_data_model.models.models import (
    AcquisitionSession,
    Alignment,
    Average,
    CTFMetadata,
    Instrument,
    PointMatrixSet3D,
    Scale,
    TiltSeries,
    Tomogram,
)
from cets_data_model.utils.yaml_io import load_cets_object, load_cets_objects
from scipion.cli import main

DATA_DIR = Path(__file__).parent / "data"
IMPORTED_TS_DB = "imported_tilt_series.sqlite"
ALIGNED_TS_DB = "aligned_tilt_series.sqlite"
CTF_DB = "ctftomoseries.sqlite"
TOMO_DB = "tomograms_bin8_no_fid.sqlite"
COORDS_DB = "coordinates3d.sqlite"
SUBTOMO_DB = "subtomograms.sqlite"
SQLITE_FIXTURES = [
    IMPORTED_TS_DB,
    ALIGNED_TS_DB,
    CTF_DB,
    TOMO_DB,
    COORDS_DB,
    SUBTOMO_DB,
]
TS_IDS_FILE = DATA_DIR / "ts_ids.txt"
COORDS_PIXEL_SIZE_FILE = DATA_DIR / "coordinates_pixel_size.txt"
UNBINNED_PIXEL_SIZE_FILE = DATA_DIR / "unbinned_pixel_size.txt"

# Tomogram in which the coordinates and subtomograms were picked
PICKED_TS_ID = "TS_03"
# Box (voxels) of the subtomograms created in the temporary project
SUBTOMO_BOX = 4
# Tolerance of the comparisons of values in Å/voxels read from files and computed
ATOL = 1e-6


def read_ts_ids(path: Path = TS_IDS_FILE) -> List[str]:
    """The tsIds of a text file, one per line (blank lines are skipped)."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_pixel_size(path: Path) -> float:
    """The pixel size (Å) of a text file holding only that value."""
    return float(path.read_text().strip())


def read_set(db_path: Path, prefix: str = "") -> List[Dict[str, Any]]:
    """Reads the items of a Scipion set (tables <prefix>Classes / <prefix>Objects) as
    dicts {attribute: value}, in the order they are stored. Independent of the converters,
    to get the values expected in the CETS files."""
    with contextlib.closing(
        sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    ) as conn:
        columns = dict(
            conn.execute(
                f'SELECT column_name, label_property FROM "{prefix}Classes"'
            ).fetchall()
        )
        cursor = conn.execute(f'SELECT * FROM "{prefix}Objects" ORDER BY id')
        names = [d[0] for d in cursor.description]
        return [
            {columns.get(name, name): value for name, value in zip(names, row)}
            for row in cursor.fetchall()
        ]


def write_sparse_mrc(
    path: Path, shape: Tuple[int, int, int], voxel_size: float
) -> None:
    """Creates an MRC file of the given shape (z, y, x) without writing its data: the file
    is sparse, so even the size of a real tilt-series takes no disk space."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new_mmap(path, shape=shape, mrc_mode=0, overwrite=True) as mrc:
        mrc.voxel_size = voxel_size


def origin_shifts(origin_matrix: str) -> List[float]:
    """Shifts (x, y, z), in Å, of a Scipion origin matrix."""
    matrix = ast.literal_eval(origin_matrix)
    return [float(matrix[i][3]) for i in range(3)]


def tomo_shape(tomo_row: Dict[str, Any]) -> Tuple[int, int, int]:
    """Shape (z, y, x) of a tomogram with Scipion's default origin (its center): its
    shifts are -dimension / 2 * sampling rate."""
    x, y, z = (
        round(-2 * shift / float(tomo_row["_samplingRate"]))
        for shift in origin_shifts(tomo_row["_origin._matrix"])
    )
    return z, y, x


def array_to_physical_scale(transformations: Optional[List[Any]]) -> List[float]:
    """The scale of the array_to_physical transformation of a CETS image."""
    scales = [
        t
        for t in transformations or []
        if isinstance(t, Scale) and t.name == "array_to_physical"
    ]
    if len(scales) != 1:
        raise AssertionError(f"Expected one array_to_physical scale, found {scales}")
    return list(scales[0].scale)


def run(args: List[str]) -> Tuple[int, str, str]:
    """Runs cets-scipion with the given arguments; returns the exit code and what it
    printed to stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


class CetsScipionCliFixturesTest(TestCase):
    """The temporary project is built once: the tests only read it and write their output
    to their own directory."""

    tmp_dir: Path
    runs: Path
    ts_ids: List[str]
    coords_pixel_size: float
    unbinned_pixel_size: float

    @classmethod
    def setUpClass(cls) -> None:
        cls.ts_ids = read_ts_ids()
        cls.coords_pixel_size = read_pixel_size(COORDS_PIXEL_SIZE_FILE)
        cls.unbinned_pixel_size = read_pixel_size(UNBINNED_PIXEL_SIZE_FILE)
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="cets_scipion_cli_fixtures"))
        prj = cls.tmp_dir / "project"
        cls.runs = prj / "Runs" / "000001_Fixtures"
        cls.runs.mkdir(parents=True)
        for name in SQLITE_FIXTURES:
            shutil.copy(DATA_DIR / name, cls.runs / name)
        # Tilt-series stacks, with the dimensions of the set (_firstDim: x, y, n)
        for ts in read_set(DATA_DIR / IMPORTED_TS_DB):
            x, y, n = (int(v) for v in ts["_firstDim"].split(","))
            stack = read_set(DATA_DIR / IMPORTED_TS_DB, f"{ts['_tsId']}_")[0]
            write_sparse_mrc(
                prj / stack["_filename"], (n, y, x), float(ts["_samplingRate"])
            )
        # Tomograms
        for tomo in read_set(DATA_DIR / TOMO_DB):
            write_sparse_mrc(
                prj / tomo["_filename"], tomo_shape(tomo), float(tomo["_samplingRate"])
            )
        # Subtomograms
        for subtomo in read_set(DATA_DIR / SUBTOMO_DB):
            write_sparse_mrc(
                prj / subtomo["_filename"],
                (SUBTOMO_BOX,) * 3,
                float(subtomo["_samplingRate"]),
            )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def setUp(self) -> None:
        self.out_dir = Path(tempfile.mkdtemp(dir=self.tmp_dir, prefix="out"))
        self.yaml = self.out_dir / "cets.yaml"

    def db(self, name: str) -> str:
        return str(self.runs / name)

    def assert_run(self, args: List[str], code: int = 0) -> str:
        """Runs cets-scipion, checks its exit code and returns what it printed to stderr."""
        result, out, err = run(args)
        self.assertEqual(result, code, msg=f"stdout:\n{out}\nstderr:\n{err}")
        return err

    # ------------------------------------------------------------------ fixtures
    def test_fixture_metadata_files(self) -> None:
        """The text files agree with the sets: the tilt-series of the dataset, their pixel
        size and that of the binned tomograms."""
        self.assertEqual(len(self.ts_ids), 2)
        self.assertEqual(len(set(self.ts_ids)), len(self.ts_ids))
        for db in (IMPORTED_TS_DB, ALIGNED_TS_DB, CTF_DB, TOMO_DB):
            with self.subTest(db=db):
                ids = [row["_tsId"] for row in read_set(DATA_DIR / db)]
                self.assertCountEqual(ids, self.ts_ids)
        for ts in read_set(DATA_DIR / IMPORTED_TS_DB):
            self.assertAlmostEqual(float(ts["_samplingRate"]), self.unbinned_pixel_size)
        for tomo in read_set(DATA_DIR / TOMO_DB):
            self.assertAlmostEqual(float(tomo["_samplingRate"]), self.coords_pixel_size)
        # The tomograms are binned by an integer factor
        binning = self.coords_pixel_size / self.unbinned_pixel_size
        self.assertAlmostEqual(binning, round(binning))

    # ------------------------------------------------------------------ tilt-series
    def _check_tilt_series(self, aligned: bool, with_ctf: bool) -> None:
        ts_db = ALIGNED_TS_DB if aligned else IMPORTED_TS_DB
        tilt_series = load_cets_objects(self.yaml, "tilt_series", TiltSeries)
        alignments = load_cets_objects(self.yaml, "alignment", Alignment)
        self.assertCountEqual([ts.id for ts in tilt_series], self.ts_ids)
        self.assertEqual(len(alignments), len(tilt_series))
        ctf_series = {row["_tsId"]: row["id"] for row in read_set(DATA_DIR / CTF_DB)}
        for ts, alignment in zip(tilt_series, alignments):
            with self.subTest(ts_id=ts.id):
                images = ts.images or []
                nominal_rows = read_set(DATA_DIR / IMPORTED_TS_DB, f"{ts.id}_")
                ts_rows = read_set(DATA_DIR / ts_db, f"{ts.id}_")
                self.assertEqual(len(images), len(ts_rows))
                # Sections are 0-based, Scipion's _index 1-based
                self.assertEqual(
                    sorted(ti.section for ti in images if ti.section is not None),
                    sorted(int(row["_index"]) - 1 for row in ts_rows),
                )
                for ti in images:
                    np.testing.assert_allclose(
                        array_to_physical_scale(ti.coordinate_transformations),
                        [self.unbinned_pixel_size] * 2,
                        atol=ATOL,
                    )
                nominal = {int(r["_index"]) - 1: r["_tiltAngle"] for r in nominal_rows}
                self.assertEqual(
                    {ti.section: ti.nominal_tilt_angle for ti in images}, nominal
                )
                projections = alignment.projection_alignments or []
                if aligned:
                    refined = sorted(float(row["_tiltAngle"]) for row in ts_rows)
                    self.assertEqual(
                        sorted(
                            p.refined_tilt_angle
                            for p in projections
                            if p.refined_tilt_angle is not None
                        ),
                        refined,
                    )
                self.assertEqual(len(projections), len(images))
                if with_ctf:
                    # Each tilt-image has the CTF of its acquisition order
                    ctfs = {
                        int(row["_acqOrder"]): row["_defocusU"]
                        for row in read_set(
                            DATA_DIR / CTF_DB, f"id{ctf_series[ts.id]}_"
                        )
                    }
                    self.assertEqual(
                        {
                            ti.acquisition_order: (
                                ti.ctf_metadata.defocus_u if ti.ctf_metadata else None
                            )
                            for ti in images
                        },
                        ctfs,
                    )
                else:
                    self.assertTrue(all(ti.ctf_metadata is None for ti in images))
        # All the tilt-series were acquired with the same microscope and settings
        instrument = load_cets_object(self.yaml, "instrument", Instrument)
        session = load_cets_object(self.yaml, "acquisition_session", AcquisitionSession)
        first = read_set(DATA_DIR / ts_db)[0]
        self.assertEqual(instrument.voltage, first["_acquisition._voltage"])
        self.assertEqual(session.instrument_id, instrument.id)
        self.assertEqual(session.tilt_axis_angle, first["_acquisition._tiltAxisAngle"])
        self.assertEqual(
            session.spherical_aberration, first["_acquisition._sphericalAberration"]
        )
        self.assertEqual(
            session.amplitude_contrast, first["_acquisition._amplitudeContrast"]
        )
        self.assertTrue(
            all(ts.acquisition_session_id == session.id for ts in tilt_series)
        )

    def test_tilt_series_aligned_with_nominal_and_ctf(self) -> None:
        err = self.assert_run(
            [
                "tilt-series",
                "to-cets",
                "--ts-db",
                self.db(ALIGNED_TS_DB),
                "--nominal-ts-db",
                self.db(IMPORTED_TS_DB),
                "--ctf-db",
                self.db(CTF_DB),
                "-o",
                str(self.yaml),
            ]
        )
        self.assertEqual(err, "")
        self._check_tilt_series(aligned=True, with_ctf=True)

    def test_tilt_series_imported(self) -> None:
        self.assert_run(
            [
                "tilt-series",
                "to-cets",
                "--ts-db",
                self.db(IMPORTED_TS_DB),
                "-o",
                str(self.yaml),
            ]
        )
        self._check_tilt_series(aligned=False, with_ctf=False)

    def test_tilt_series_aligned_without_nominal(self) -> None:
        """Scipion only keeps the refined tilt angles of an aligned set: without
        --nominal-ts-db, the nominal ones are left empty."""
        self.assert_run(
            [
                "tilt-series",
                "to-cets",
                "--ts-db",
                self.db(ALIGNED_TS_DB),
                "-o",
                str(self.yaml),
            ]
        )
        for ts in load_cets_objects(self.yaml, "tilt_series", TiltSeries):
            self.assertTrue(
                all(ti.nominal_tilt_angle is None for ti in ts.images or [])
            )
        for alignment in load_cets_objects(self.yaml, "alignment", Alignment):
            self.assertTrue(
                all(
                    p.refined_tilt_angle is not None
                    for p in alignment.projection_alignments or []
                )
            )

    # ------------------------------------------------------------------ ctf
    def test_ctf_each_tilt_series(self) -> None:
        series = {row["_tsId"]: row["id"] for row in read_set(DATA_DIR / CTF_DB)}
        for ts_id in self.ts_ids:
            with self.subTest(ts_id=ts_id):
                yaml_file = self.out_dir / f"ctf_{ts_id}.yaml"
                self.assert_run(
                    [
                        "ctf",
                        "to-cets",
                        "--ctf-db",
                        self.db(CTF_DB),
                        "--ts-id",
                        ts_id,
                        "-o",
                        str(yaml_file),
                    ]
                )
                ctfs = load_cets_objects(yaml_file, "ctf", CTFMetadata)
                rows = read_set(DATA_DIR / CTF_DB, f"id{series[ts_id]}_")
                # In the order stored in Scipion
                self.assertEqual(
                    [(c.defocus_u, c.defocus_v, c.defocus_angle) for c in ctfs],
                    [
                        (r["_defocusU"], r["_defocusV"], r["_defocusAngle"])
                        for r in rows
                    ],
                )

    def test_ctf_needs_a_valid_ts_id(self) -> None:
        args = ["ctf", "to-cets", "--ctf-db", self.db(CTF_DB), "-o", str(self.yaml)]
        # The set has the CTFs of several tilt-series: --ts-id is required
        err = self.assert_run(args, code=1)
        for ts_id in self.ts_ids:
            self.assertIn(ts_id, err)
        self.assertIn("TS_99", self.assert_run(args + ["--ts-id", "TS_99"], code=1))
        self.assertFalse(self.yaml.exists())

    # ------------------------------------------------------------------ tomogram
    def test_tomograms(self) -> None:
        self.assert_run(
            ["tomogram", "to-cets", "--tomo-db", self.db(TOMO_DB), "-o", str(self.yaml)]
        )
        tomos = load_cets_objects(self.yaml, "tomogram", Tomogram)
        self.assertCountEqual([t.id for t in tomos], self.ts_ids)
        rows = {row["_tsId"]: row for row in read_set(DATA_DIR / TOMO_DB)}
        for tomo in tomos:
            with self.subTest(tomo_id=tomo.id):
                self.assertEqual(
                    (tomo.depth, tomo.height, tomo.width), tomo_shape(rows[tomo.id])
                )
                np.testing.assert_allclose(
                    array_to_physical_scale(tomo.coordinate_transformations),
                    [self.coords_pixel_size] * 3,
                    atol=ATOL,
                )
                self.assertIs(tomo.ctf_corrected, False)

    # ------------------------------------------------------------------ coordinates
    def _coords_args(self, *extra: str, output: Optional[Path] = None) -> List[str]:
        return [
            "coordinates",
            "to-cets",
            "--coords-db",
            self.db(COORDS_DB),
            "--tomo-db",
            self.db(TOMO_DB),
            "-o",
            str(output or self.yaml),
            *extra,
        ]

    def _expected_corner_coords(self, x: str, y: str, z: str, db: str) -> List[Any]:
        """Coordinates of a set referred to the corner of the tomogram of PICKED_TS_ID."""
        tomo = next(
            row for row in read_set(DATA_DIR / TOMO_DB) if row["_tsId"] == PICKED_TS_ID
        )
        offset = [
            -shift / float(tomo["_samplingRate"])
            for shift in origin_shifts(tomo["_origin._matrix"])
        ]
        coords = [
            [float(row[x]), float(row[y]), float(row[z])]
            for row in read_set(DATA_DIR / db)
        ]
        return (np.array(coords) + offset).tolist()

    def test_coordinates(self) -> None:
        self.assert_run(self._coords_args("--pixel-size", str(self.coords_pixel_size)))
        # Only TS_03 has coordinates; they are oriented
        points = load_cets_object(self.yaml, "coordinates", PointMatrixSet3D)
        rows = read_set(DATA_DIR / COORDS_DB)
        self.assertEqual(points.source_tomogram_id, PICKED_TS_ID)
        self.assertEqual(len(points.origin3D or []), len(rows))
        self.assertEqual(len(points.matrix3D or []), len(rows))
        self.assertEqual(points.scores, [row["_score"] for row in rows])
        np.testing.assert_allclose(
            points.origin3D or [],
            self._expected_corner_coords("_x", "_y", "_z", COORDS_DB),
            atol=ATOL,
        )
        # Referred to the corner: inside the tomogram
        tomo = next(
            row for row in read_set(DATA_DIR / TOMO_DB) if row["_tsId"] == PICKED_TS_ID
        )
        z, y, x = tomo_shape(tomo)
        origins = np.array(points.origin3D)
        self.assertTrue(np.all(origins >= 0))
        self.assertTrue(np.all(origins <= [x, y, z]))
        # Orientations: the inverse (transpose) of the Scipion rotations
        for matrix, row in zip(points.matrix3D or [], rows):
            scipion = np.array(ast.literal_eval(row["_eulerMatrix._matrix"]))[:3, :3]
            np.testing.assert_allclose(matrix, scipion.T, atol=1e-9)
        np.testing.assert_allclose(
            array_to_physical_scale(points.coordinate_transformations),
            [self.coords_pixel_size] * 3,
            atol=ATOL,
        )

    def test_coordinates_without_pixel_size(self) -> None:
        self.assert_run(self._coords_args())
        points = load_cets_object(self.yaml, "coordinates", PointMatrixSet3D)
        self.assertFalse(points.coordinate_transformations)

    def test_coordinates_tomo_ids(self) -> None:
        # The tomogram with coordinates
        self.assert_run(self._coords_args("--tomo-id", PICKED_TS_ID))
        self.assertEqual(
            load_cets_object(
                self.yaml, "coordinates", PointMatrixSet3D
            ).source_tomogram_id,
            PICKED_TS_ID,
        )
        self.yaml.unlink()
        # All the tilt-series of the dataset: the others have no coordinates
        tomo_id_args = [arg for ts_id in self.ts_ids for arg in ("--tomo-id", ts_id)]
        err = self.assert_run(self._coords_args(*tomo_id_args), code=1)
        self.assertIn("TS_54", err)
        self.assertFalse(self.yaml.exists())
        # Unknown tomogram
        self.assert_run(self._coords_args("--tomo-id", "TS_99"), code=1)
        self.assertFalse(self.yaml.exists())

    # ------------------------------------------------------------------ subtomograms
    def _subtomo_args(self, *extra: str) -> List[str]:
        return [
            "subtomograms",
            "to-cets",
            "--subtomo-db",
            self.db(SUBTOMO_DB),
            "-o",
            str(self.yaml),
            *extra,
        ]

    def test_subtomograms(self) -> None:
        self.assert_run(
            self._subtomo_args(
                "--tomo-db",
                self.db(TOMO_DB),
                "--coords-pixel-size",
                str(self.coords_pixel_size),
            )
        )
        points = load_cets_object(self.yaml, "coordinates", PointMatrixSet3D)
        average = load_cets_object(self.yaml, "average", Average)
        rows = read_set(DATA_DIR / SUBTOMO_DB)
        self.assertEqual(points.source_tomogram_id, PICKED_TS_ID)
        np.testing.assert_allclose(
            points.origin3D or [],
            self._expected_corner_coords(
                "_coordinate._x", "_coordinate._y", "_coordinate._z", SUBTOMO_DB
            ),
            atol=ATOL,
        )
        np.testing.assert_allclose(
            array_to_physical_scale(points.coordinate_transformations),
            [self.coords_pixel_size] * 3,
            atol=ATOL,
        )
        maps = average.particle_maps or []
        self.assertEqual([m.coord_index for m in maps], list(range(len(rows))))
        self.assertTrue(all(m.source_annotation_id == points.id for m in maps))
        self.assertEqual(
            [(m.depth, m.height, m.width) for m in maps],
            [(SUBTOMO_BOX,) * 3] * len(rows),
        )

    def test_subtomograms_match_coordinates(self) -> None:
        """The subtomograms were extracted from the coordinates: both commands give the
        same positions and orientations, and the default origin (the center of the
        tomogram, read from its file) is the one stored in the set of tomograms."""
        coords_yaml = self.out_dir / "coords.yaml"
        self.assert_run(self._coords_args(output=coords_yaml))
        self.assert_run(self._subtomo_args("--tomo-id", PICKED_TS_ID))
        coords = load_cets_object(coords_yaml, "coordinates", PointMatrixSet3D)
        subtomos = load_cets_object(self.yaml, "coordinates", PointMatrixSet3D)
        np.testing.assert_allclose(
            subtomos.origin3D or [], coords.origin3D or [], atol=ATOL
        )
        np.testing.assert_allclose(
            subtomos.matrix3D or [], coords.matrix3D or [], atol=1e-9
        )

    def test_subtomograms_unknown_tomo_id(self) -> None:
        err = self.assert_run(self._subtomo_args("--tomo-id", "TS_99"), code=1)
        self.assertIn(PICKED_TS_ID, err)  # The available ones are listed
        self.assertFalse(self.yaml.exists())

    # ------------------------------------------------------------------ command line
    def test_output_dir_is_created(self) -> None:
        nested = self.out_dir / "a" / "b" / "ctf.yaml"
        self.assert_run(
            [
                "ctf",
                "to-cets",
                "--ctf-db",
                self.db(CTF_DB),
                "--ts-id",
                self.ts_ids[0],
                "-o",
                str(nested),
            ]
        )
        self.assertTrue(nested.exists())

    def test_input_files_are_not_modified(self) -> None:
        """The command only reads the Scipion sets."""
        before = {name: (self.runs / name).read_bytes() for name in SQLITE_FIXTURES}
        self.assert_run(
            self._subtomo_args(
                "--tomo-db",
                self.db(TOMO_DB),
                "--coords-pixel-size",
                str(self.coords_pixel_size),
            )
        )
        self.assert_run(self._coords_args())
        after = {name: (self.runs / name).read_bytes() for name in SQLITE_FIXTURES}
        self.assertEqual(before, after)

    def test_invalid_arguments(self) -> None:
        """Usage errors exit with code 2 (argparse) before running any conversion."""
        invalid: List[List[str]] = [
            [],
            ["tomogram"],
            ["tomogram", "from-cets"],
            ["tomogram", "to-cets", "-o", str(self.yaml)],  # --tomo-db is required
            ["coordinates", "to-cets", "--coords-db", self.db(COORDS_DB)],
            ["ctf", "to-cets", "--ctf-db", self.db(CTF_DB), "--pixel-size", "1"],
        ]
        for args in invalid:
            with self.subTest(args=args):
                with self.assertRaises(SystemExit) as ctx:
                    run(args)
                self.assertEqual(ctx.exception.code, 2)
        self.assertFalse(self.yaml.exists())

    def test_help(self) -> None:
        for args in (["--help"], ["subtomograms", "to-cets", "--help"]):
            with self.subTest(args=args):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    with self.assertRaises(SystemExit) as ctx:
                        main(args)
                self.assertEqual(ctx.exception.code, 0)
                self.assertIn("cets-scipion", out.getvalue())

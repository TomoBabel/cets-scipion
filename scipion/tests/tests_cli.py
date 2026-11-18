# -*- coding: utf-8 -*-
import contextlib
import io
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List
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
    PointSet3D,
    TiltSeries,
    Tomogram,
)
from cets_data_model.utils.yaml_io import load_cets_object, load_cets_objects
from scipion.cli import main

TS_ID = "TS_01"
TS_ID_2 = "TS_02"
PIXEL_SIZE = 2.0
NOMINAL = [-3.0, 0.0, 3.0]
REFINED = [-2.91, 0.04, 3.12]
TILT_AXIS_ANGLE = 85.3
DEFOCUS_U = [30010.0, 30020.0, 30030.0]
# Tomogram of 8 x 8 x 4 voxels with Scipion's default origin (its center)
TOMO_SHAPE = (4, 8, 8)  # z, y, x
TOMO_ORIGIN = [-8.0, -8.0, -4.0]  # Å: -dimension / 2 * voxel size
COORDS = [(1.0, 2.0, 0.5), (-1.0, -2.0, 1.0)]
CORNER_COORDS = [[5.0, 6.0, 2.5], [3.0, 2.0, 3.0]]  # + (4, 4, 2) voxels
IDENTITY_4 = str(np.eye(4).tolist())


def rotation_z(degrees: float) -> np.ndarray:
    a = np.deg2rad(degrees)
    return np.array(
        [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )


# Orientations of the coordinates as Scipion stores them (particle -> reference)
SCIPION_ROTATIONS = [rotation_z(30.0), rotation_z(-75.0)]


def run(args: List[str]) -> int:
    """Runs cets-scipion with the given arguments, hiding its messages."""
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return main(args)


def write_mrc(path: Path, shape: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path) as mrc:
        mrc.set_data(np.zeros(shape, dtype=np.float32))
        mrc.voxel_size = PIXEL_SIZE


def write_tables(db_path: Path, rows: List[Dict[str, Any]], prefix: str = "") -> Path:
    """Writes the tables <prefix>Classes / <prefix>Objects of a Scipion set: the table
    Classes maps the attributes (the keys of ``rows``) to the columns of the table
    Objects."""
    labels = list(rows[0])
    columns = {label: f"c{i:02d}" for i, label in enumerate(labels, start=1)}
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f'CREATE TABLE "{prefix}Classes" (id INTEGER PRIMARY KEY, '
            f"label_property TEXT, column_name TEXT, class_name TEXT)"
        )
        conn.executemany(
            f'INSERT INTO "{prefix}Classes" (label_property, column_name, class_name) '
            f"VALUES (?, ?, ?)",
            [(label, column, "String") for label, column in columns.items()],
        )
        conn.execute(
            f'CREATE TABLE "{prefix}Objects" (id INTEGER PRIMARY KEY, '
            f"{', '.join(columns.values())})"
        )
        for row in rows:
            conn.execute(
                f'INSERT INTO "{prefix}Objects" ({", ".join(columns.values())}) '
                f"VALUES ({', '.join('?' * len(labels))})",
                [row[label] for label in labels],
            )
    conn.close()
    return db_path


class CetsScipionCliTest(TestCase):
    def setUp(self) -> None:
        # Scipion project: the paths in the sets are relative to it, and the sets are in
        # <project>/Runs/<protocol>/
        self.dir = Path(tempfile.mkdtemp(prefix="cets_scipion_cli"))
        self.prj = self.dir / "project"
        self.yaml = self.dir / "out" / "cets.yaml"
        self.stack = "Runs/000001_ProtImportTs/extra/TS_01.mrcs"
        write_mrc(self.prj / self.stack, (len(NOMINAL), 8, 8))
        self.tomo = "Runs/000004_ProtImportTomograms/extra/TS_01.mrc"
        write_mrc(self.prj / self.tomo, TOMO_SHAPE)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _db(self, protocol: str, name: str) -> Path:
        path = self.prj / "Runs" / protocol / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _ts_db(
        self, protocol: str, aligned: bool, with_ctf_corrected: bool = True
    ) -> Path:
        db = self._db(protocol, "tiltseries.sqlite")
        # As in Scipion, the acquisition of the tilt-series (with the tilt axis angle) is
        # in its row of the set; the tilt-images only have a copy of some of its values.
        ts_row: Dict[str, Any] = {
            "_tsId": TS_ID,
            "_acquisition._voltage": 300.0,
            "_acquisition._sphericalAberration": 2.7,
            "_acquisition._amplitudeContrast": 0.07,
            "_acquisition._tiltAxisAngle": TILT_AXIS_ANGLE,
        }
        if with_ctf_corrected:  # Older sets do not have it
            ts_row["_ctfCorrected"] = 0
        write_tables(db, [ts_row])
        angles = REFINED if aligned else NOMINAL
        rows = []
        for i, angle in enumerate(angles):
            row: Dict[str, Any] = {
                "_tsId": TS_ID,
                "_filename": self.stack,
                "_index": i + 1,
                "_acqOrder": i,
                "_tiltAngle": angle,
                "_acquisition._accumDose": 3.0 * i,
                "_acquisition._voltage": 300.0,
                "_acquisition._sphericalAberration": 2.7,
                "_acquisition._amplitudeContrast": 0.07,
            }
            if aligned:
                row["_transform._matrix"] = str(
                    [[1.0, 0.0, 1.5 * i], [0.0, 1.0, -i], [0.0, 0.0, 1.0]]
                )
            rows.append(row)
        return write_tables(db, rows, prefix=f"{TS_ID}_")

    def _ctf_db(self) -> Path:
        db = self._db("000003_ProtImportTsCTF", "ctftomoseries.sqlite")
        write_tables(db, [{"_tsId": TS_ID}, {"_tsId": TS_ID_2}])
        for obj_id in (1, 2):
            # Stored in reverse order: matched by acquisition order
            write_tables(
                db,
                [
                    {
                        "_defocusU": u,
                        "_defocusV": u - 100,
                        "_defocusAngle": 10.0,
                        "_acqOrder": i,
                    }
                    for i, u in reversed(list(enumerate(DEFOCUS_U)))
                ],
                prefix=f"id{obj_id}_",
            )
        return db

    def _tomo_db(self) -> Path:
        origin = np.eye(4)
        origin[:3, 3] = TOMO_ORIGIN
        return write_tables(
            self._db("000004_ProtImportTomograms", "tomograms.sqlite"),
            [
                {
                    "_tsId": TS_ID,
                    "_filename": self.tomo,
                    "_ctfCorrected": 0,
                    "_samplingRate": PIXEL_SIZE,
                    "_origin._matrix": str(origin.tolist()),
                }
            ],
        )

    def _coords_db(self, oriented: bool = False) -> Path:
        return write_tables(
            self._db("000005_ProtImportCoordinates3D", "coordinates3d.sqlite"),
            [
                {
                    "_x": x,
                    "_y": y,
                    "_z": z,
                    "_tomoId": TS_ID,
                    "_eulerMatrix._matrix": (
                        str(np.pad(rotation, (0, 1)).tolist())
                        if oriented
                        else IDENTITY_4
                    ),
                    "_score": 0.5,
                }
                for (x, y, z), rotation in zip(COORDS, SCIPION_ROTATIONS)
            ],
        )

    def _subtomo_db(self) -> Path:
        rows = []
        for i, (x, y, z) in enumerate(COORDS):
            subtomo = f"Runs/000006_ProtExtractSubtomos/extra/{i}.mrc"
            write_mrc(self.prj / subtomo, (4, 4, 4))
            rows.append(
                {
                    "_filename": subtomo,
                    "_coordinate._x": x,
                    "_coordinate._y": y,
                    "_coordinate._z": z,
                    "_coordinate._eulerMatrix._matrix": IDENTITY_4,
                    "_transform._matrix": IDENTITY_4,
                    "_coordinate._tomoId": TS_ID,
                    "_volName": self.tomo,
                }
            )
        return write_tables(
            self._db("000006_ProtExtractSubtomos", "subtomograms.sqlite"), rows
        )

    def test_tilt_series_to_cets(self) -> None:
        aligned = self._ts_db("000002_ProtImodImportTransformationMatrix", True)
        nominal = self._ts_db("000001_ProtImportTs", False)
        code = run(
            [
                "tilt-series",
                "to-cets",
                "--ts-db",
                str(aligned),
                "--nominal-ts-db",
                str(nominal),
                "--ctf-db",
                str(self._ctf_db()),
                "-o",
                str(self.yaml),
            ]
        )
        self.assertEqual(code, 0)
        ts = load_cets_object(self.yaml, "tilt_series", TiltSeries)
        alignment = load_cets_object(self.yaml, "alignment", Alignment)
        images = ts.images or []
        self.assertEqual([ti.nominal_tilt_angle for ti in images], NOMINAL)
        self.assertEqual(
            [p.refined_tilt_angle for p in alignment.projection_alignments or []],
            REFINED,
        )
        self.assertEqual(
            [ti.ctf_metadata.defocus_u for ti in images if ti.ctf_metadata], DEFOCUS_U
        )
        instrument = load_cets_object(self.yaml, "instrument", Instrument)
        session = load_cets_object(self.yaml, "acquisition_session", AcquisitionSession)
        self.assertEqual(instrument.voltage, 300.0)
        self.assertEqual(session.instrument_id, instrument.id)
        # Read from the tilt-series row of the set (TomoAcquisition)
        self.assertEqual(session.tilt_axis_angle, TILT_AXIS_ANGLE)
        self.assertIs(ts.ctf_corrected, False)

    def test_tilt_series_not_aligned(self) -> None:
        db = self._ts_db("000001_ProtImportTs", False)
        code = run(["tilt-series", "to-cets", "--ts-db", str(db), "-o", str(self.yaml)])
        self.assertEqual(code, 0)
        ts = load_cets_object(self.yaml, "tilt_series", TiltSeries)
        self.assertEqual([ti.nominal_tilt_angle for ti in ts.images or []], NOMINAL)
        self.assertTrue(all(ti.ctf_metadata is None for ti in ts.images or []))

    def test_tilt_series_without_ctf_corrected(self) -> None:
        """Sets without the _ctfCorrected column are converted, leaving it unknown."""
        db = self._ts_db("000001_ProtImportTs", False, with_ctf_corrected=False)
        code = run(["tilt-series", "to-cets", "--ts-db", str(db), "-o", str(self.yaml)])
        self.assertEqual(code, 0)
        self.assertIsNone(
            load_cets_object(self.yaml, "tilt_series", TiltSeries).ctf_corrected
        )

    def test_ctf_to_cets(self) -> None:
        ctf_db = str(self._ctf_db())
        to_cets = ["ctf", "to-cets", "--ctf-db", ctf_db, "-o", str(self.yaml)]
        # Several tilt-series in the set: --ts-id is needed
        self.assertEqual(run(to_cets), 1)
        self.assertEqual(run(to_cets + ["--ts-id", "TS_99"]), 1)
        self.assertEqual(run(to_cets + ["--ts-id", TS_ID_2]), 0)
        ctfs = load_cets_objects(self.yaml, "ctf", CTFMetadata)
        # In the order stored in Scipion
        self.assertEqual([c.defocus_u for c in ctfs], DEFOCUS_U[::-1])

    def test_tomogram_to_cets(self) -> None:
        code = run(
            [
                "tomogram",
                "to-cets",
                "--tomo-db",
                str(self._tomo_db()),
                "-o",
                str(self.yaml),
            ]
        )
        self.assertEqual(code, 0)
        tomos = load_cets_objects(self.yaml, "tomogram", Tomogram)
        self.assertEqual([t.id for t in tomos], [TS_ID])
        self.assertEqual((tomos[0].depth, tomos[0].height, tomos[0].width), TOMO_SHAPE)

    def test_coordinates_to_cets(self) -> None:
        args = [
            "coordinates",
            "to-cets",
            "--coords-db",
            str(self._coords_db()),
            "--tomo-db",
            str(self._tomo_db()),
            "--pixel-size",
            str(PIXEL_SIZE),
            "-o",
            str(self.yaml),
        ]
        self.assertEqual(run(args), 0)
        points = load_cets_object(self.yaml, "coordinates", PointSet3D)
        self.assertEqual(points.source_tomogram_id, TS_ID)
        self.assertEqual(points.origin3D, CORNER_COORDS)
        self.assertEqual(points.scores, [0.5, 0.5])
        self.assertEqual(len(points.coordinate_transformations or []), 1)
        # Unknown tomogram
        self.assertEqual(run(args + ["--tomo-id", "TS_99"]), 1)

    def test_oriented_coordinates_to_cets(self) -> None:
        """Oriented coordinates keep their orientation, from the reference to the particle:
        the inverse of the Scipion matrix."""
        args = [
            "coordinates",
            "to-cets",
            "--coords-db",
            str(self._coords_db(oriented=True)),
            "--tomo-db",
            str(self._tomo_db()),
            "-o",
            str(self.yaml),
        ]
        self.assertEqual(run(args), 0)
        points = load_cets_object(self.yaml, "coordinates", PointMatrixSet3D)
        self.assertEqual(points.origin3D, CORNER_COORDS)
        for matrix, scipion_rotation in zip(points.matrix3D or [], SCIPION_ROTATIONS):
            np.testing.assert_allclose(matrix, scipion_rotation.T, atol=1e-12)

    def test_subtomograms_to_cets(self) -> None:
        args = [
            "subtomograms",
            "to-cets",
            "--subtomo-db",
            str(self._subtomo_db()),
            "--tomo-id",
            TS_ID,
            "-o",
            str(self.yaml),
        ]
        # Without --tomo-db: the origin is the center of the tomogram (_volName)
        self.assertEqual(run(args), 0)
        points = load_cets_object(self.yaml, "coordinates", PointSet3D)
        average = load_cets_object(self.yaml, "average", Average)
        self.assertEqual(points.origin3D, CORNER_COORDS)
        maps = average.particle_maps or []
        self.assertEqual([m.coord_index for m in maps], [0, 1])
        self.assertTrue(all(m.source_annotation_id == points.id for m in maps))
        self.assertEqual(
            run(args + ["--tomo-db", str(self._tomo_db()), "--coords-pixel-size", "2"]),
            0,
        )
        points = load_cets_object(self.yaml, "coordinates", PointSet3D)
        self.assertEqual(points.origin3D, CORNER_COORDS)
        self.assertEqual(len(points.coordinate_transformations or []), 1)

    def test_errors_give_exit_code_1(self) -> None:
        missing = self.prj / "Runs" / "000009_Missing" / "missing.sqlite"
        for args in (
            ["tilt-series", "to-cets", "--ts-db", str(missing)],
            ["ctf", "to-cets", "--ctf-db", str(missing)],
            ["tomogram", "to-cets", "--tomo-db", str(missing)],
            [
                "coordinates",
                "to-cets",
                "--coords-db",
                str(self._coords_db()),
                "--tomo-db",
                str(missing),
            ],
            ["subtomograms", "to-cets", "--subtomo-db", str(missing)],
        ):
            with self.subTest(args=args):
                self.assertEqual(run(args + ["-o", str(self.yaml)]), 1)
                # The missing sqlite file is not created
                self.assertFalse(missing.exists())
        self.assertFalse(self.yaml.exists())

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Optional

from scipion.converters.coodinates3d import ScipionSetOfCoordinates3D

TOMO_ID = "TS_01"
# (x, y, z, score) of each coordinate, in the order they are stored
COORDS = [(10.0, 20.0, 30.0, 0.31), (40.0, 50.0, 60.0, 0.12)]
# Tomogram of 928 x 928 x 280 voxels at 5.4 Å/voxel, with Scipion's default origin (its
# center): the shifts of the origin are -dimension / 2 * sampling rate, in Å.
TOMO_SAMPLING_RATE = 5.4
TOMO_ORIGIN = (-2505.6000000000004, -2505.6000000000004, -756.0)
# COORDS referred to the corner of the tomogram (+ 464, 464, 140 voxels)
CORNER_COORDS = [[474.0, 484.0, 170.0], [504.0, 514.0, 200.0]]


def write_coords_sqlite(
    db_path: Path, with_score_column: bool = True, null_score: bool = False
) -> Path:
    """Writes a minimal Scipion SetOfCoordinates3D sqlite: the table Classes maps the
    attributes of Coordinate3D to the columns of the table Objects."""
    columns = {"_x": "c01", "_y": "c02", "_z": "c03", "_tomoId": "c04"}
    if with_score_column:
        columns["_score"] = "c05"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE Classes (id INTEGER PRIMARY KEY, label_property TEXT, "
            "column_name TEXT, class_name TEXT)"
        )
        conn.executemany(
            "INSERT INTO Classes (label_property, column_name, class_name) VALUES (?, ?, ?)",
            [(label, column, "Float") for label, column in columns.items()],
        )
        col_defs = ", ".join(f"{column} TEXT" for column in columns.values())
        conn.execute(f"CREATE TABLE Objects (id INTEGER PRIMARY KEY, {col_defs})")
        for i, (x, y, z, score) in enumerate(COORDS):
            values: list[Optional[float | str]] = [x, y, z, TOMO_ID]
            if with_score_column:
                values.append(None if null_score and i == 0 else score)
            placeholders = ", ".join("?" * len(values))
            conn.execute(
                f"INSERT INTO Objects ({', '.join(columns.values())}) "
                f"VALUES ({placeholders})",
                values,
            )
    conn.close()
    return db_path


def write_tomos_sqlite(db_path: Path) -> Path:
    """Writes a minimal Scipion SetOfTomograms sqlite with the tomogram the coordinates
    are referred to: only its tsId, sampling rate and origin."""
    columns = {"_tsId": "c01", "_samplingRate": "c02", "_origin._matrix": "c03"}
    origin_matrix = [
        [1.0, 0.0, 0.0, TOMO_ORIGIN[0]],
        [0.0, 1.0, 0.0, TOMO_ORIGIN[1]],
        [0.0, 0.0, 1.0, TOMO_ORIGIN[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE Classes (id INTEGER PRIMARY KEY, label_property TEXT, "
            "column_name TEXT, class_name TEXT)"
        )
        conn.executemany(
            "INSERT INTO Classes (label_property, column_name, class_name) VALUES (?, ?, ?)",
            [(label, column, "String") for label, column in columns.items()],
        )
        conn.execute(
            "CREATE TABLE Objects (id INTEGER PRIMARY KEY, c01 TEXT, c02 REAL, c03 TEXT)"
        )
        conn.execute(
            "INSERT INTO Objects (c01, c02, c03) VALUES (?, ?, ?)",
            (TOMO_ID, TOMO_SAMPLING_RATE, str(origin_matrix)),
        )
    conn.close()
    return db_path


class CetsScipionCoordinates3DTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / "coordinates3d.sqlite"
        self.tomo_db_path = write_tomos_sqlite(self.test_dir / "tomograms.sqlite")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_scores(self) -> None:
        """Coordinate3D._score -> PointSet3D.scores, aligned with origin3D"""
        write_coords_sqlite(self.db_path)
        points = ScipionSetOfCoordinates3D(self.db_path).scipion_to_cets(
            TOMO_ID, self.tomo_db_path
        )
        assert points is not None
        self.assertEqual(points.origin3D, CORNER_COORDS)
        self.assertEqual(points.scores, [c[3] for c in COORDS])

    def test_set_without_score_column(self) -> None:
        write_coords_sqlite(self.db_path, with_score_column=False)
        points = ScipionSetOfCoordinates3D(self.db_path).scipion_to_cets(
            TOMO_ID, self.tomo_db_path
        )
        assert points is not None
        self.assertEqual(len(points.origin3D), len(COORDS))
        self.assertIsNone(points.scores)

    def test_scores_skipped_if_any_is_missing(self) -> None:
        write_coords_sqlite(self.db_path, null_score=True)
        points = ScipionSetOfCoordinates3D(self.db_path).scipion_to_cets(
            TOMO_ID, self.tomo_db_path
        )
        assert points is not None
        self.assertIsNone(points.scores)

    def test_tomogram_not_in_tomograms_set(self) -> None:
        write_coords_sqlite(self.db_path)
        with sqlite3.connect(self.tomo_db_path) as conn:
            conn.execute("UPDATE Objects SET c01 = 'TS_99'")
        conn.close()
        with self.assertRaises(ValueError):
            ScipionSetOfCoordinates3D(self.db_path).scipion_to_cets(
                TOMO_ID, self.tomo_db_path
            )

    def test_no_coordinates_for_tomogram(self) -> None:
        write_coords_sqlite(self.db_path)
        self.assertIsNone(
            ScipionSetOfCoordinates3D(self.db_path).scipion_to_cets(
                "TS_99", self.tomo_db_path
            )
        )


if __name__ == "__main__":
    unittest.main()

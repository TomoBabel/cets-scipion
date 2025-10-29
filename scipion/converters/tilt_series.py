import ast
import os
import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np

from cets_data_model.models.models import (
    TiltImage,
    Axis,
    SpaceAxis,
    AxisUnit,
    AxisType,
    CoordinateSystem,
    CoordinateTransformation,
    Translation,
    Vector3D,
    Affine,
    Matrix3x3,
    TiltSeries,
)
from cets_data_model.utils.image_utils import get_mrc_info
from scipion.constants import (
    TS_ID,
    TILT_SERIES_FIELDS,
    FILE_NAME,
    INDEX,
    TILT_ANGLE,
    ACCUMULATED_DOSE,
    ACQUISITION_ORDER,
    TRANSFORMATION_MATRIX,
    ODD_EVEN_FN,
    CTF_CORRECTED,
)
from scipion.utils.utils import validate_file, write_ts_set_yaml
from scipion.utils.utils_sqlite import connect_db, map_classes_table, get_from_obj_tbl


class ScipionTiltSeries:
    def __init__(self, sqlite_path: os.PathLike):
        self.db_path = validate_file(sqlite_path, ".sqlite")
        self.scipion_prj_path = self._get_prj_path()

    def scipion_to_cets(
        self,
        out_directory: str | None = None,
    ) -> List[TiltSeries] | None:
        """Converts tilt-series from Scipion into CETS metadata.

        :param out_directory: name of the directory in which the tilt-series
        .yaml files (one per tilt-series) will be written.
        :type out_directory: pathlib.Path or str, optional
        """
        db_connection = connect_db(self.db_path)
        if db_connection is not None:
            with db_connection as conn:
                # Map the table Classes and get some values from the table Objects
                ts_set_class_dict = map_classes_table(conn)
                ts_ids = get_from_obj_tbl(conn, TS_ID, ts_set_class_dict)
                ctf_corrected_list = get_from_obj_tbl(
                    conn, CTF_CORRECTED, ts_set_class_dict
                )

                # Map the table Classes of the first tilt-series
                ts_class_dict = map_classes_table(
                    conn, self._get_ts_classes_tbl_name(ts_ids[0])
                )

                # Sqlite fields of the data to be read from each tilt-image
                ti_sql_fields = ", ".join(
                    [f'"{ts_class_dict[field]}"' for field in TILT_SERIES_FIELDS]
                )

                # Coordinate system
                axis_xy = Axis(
                    name=SpaceAxis.XY,
                    axis_unit=AxisUnit.pixel,
                    axis_type=AxisType.space,
                )
                coordinate_systems = CoordinateSystem(name="SCIPION", axes=[axis_xy])

                cursor = conn.cursor()
                tilt_series_list = []
                for i, ts_id in enumerate(ts_ids):
                    print(f"Loading tsId = {ts_id}")
                    # Read the tilt-images table
                    ti_list = []
                    tilt_images_table_name = self._get_ts_obj_tbl_name(ts_id)
                    query = f'SELECT {ti_sql_fields} FROM "{tilt_images_table_name}"'
                    cursor.execute(query)  # execute the query
                    for row in cursor.fetchall():
                        ti = self._ti_from_sqlite_row(
                            row, ts_class_dict, coordinate_systems
                        )
                        ti_list.append(ti)

                    # Tilt-series
                    print(cursor.fetchall())
                    ts = TiltSeries(
                        path=ti_list[-1].path,
                        ts_id=ts_id,
                        # pixel_size=pixel_size,
                        ctf_corrected=bool(ctf_corrected_list[i]),
                        images=ti_list,
                    )
                    tilt_series_list.append(ts)

                if out_directory:
                    write_ts_set_yaml(tilt_series_list, Path(out_directory))
                return tilt_series_list
        return None

    def _ti_from_sqlite_row(
        self,
        row: sqlite3.Row,
        ts_class_dict: Dict[str, str],
        coord_system: CoordinateSystem,
    ) -> TiltImage:
        # Read image info
        ts_fn = self.scipion_prj_path / row[ts_class_dict[FILE_NAME]]
        img_info = get_mrc_info(ts_fn)
        # Get the odd / even filenames
        even_fn, odd_fn = None, None
        odd_even_fn = row[ts_class_dict[ODD_EVEN_FN]]
        if odd_even_fn:
            even_fn, odd_fn = sorted(odd_even_fn.split(","))
        # Get the transformation matrix
        tr_matrix_str = row[ts_class_dict[TRANSFORMATION_MATRIX]]
        tr_matrix = np.array(ast.literal_eval(tr_matrix_str))

        # Create the tilt-image
        return TiltImage(
            ts_id=row[ts_class_dict[TS_ID]],
            path=str(ts_fn),
            even_path=even_fn,
            odd_path=odd_fn,
            acquisition_order=row[ts_class_dict[ACQUISITION_ORDER]],
            section=row[ts_class_dict[INDEX]],
            nominal_tilt_angle=row[ts_class_dict[TILT_ANGLE]],
            accumulated_dose=row[ts_class_dict[ACCUMULATED_DOSE]],
            # ctf_metadata=None,  # TODO: fill this
            width=img_info.size_x,
            height=img_info.size_y,
            coordinate_systems=[coord_system],
            coordinate_transformations=[
                self._gen_translation_transform(tr_matrix),
                self._gen_rotation_transform(tr_matrix),
            ],
        )

    @staticmethod
    def _get_ts_classes_tbl_name(ts_id: str) -> str:
        return ts_id + "_Classes"

    @staticmethod
    def _get_ts_obj_tbl_name(ts_id: str) -> str:
        return ts_id + "_Objects"

    def _get_prj_path(self) -> Path:
        orig_dir = os.getcwd()
        # Move to the project directory and get the full path
        os.chdir(
            self.db_path.parent / ".." / ".."
        )  # PathToScipionUserData/projects/ProjectName/Runs/ProtocolDir/extra/sqlite
        prj_path = os.getcwd()
        # Move to the original working dir
        os.chdir(orig_dir)
        return Path(prj_path)

    @staticmethod
    def _gen_translation_transform(transformation_matrix: np.ndarray) -> Translation:
        translation: Vector3D = [
            transformation_matrix[0, 2],
            transformation_matrix[1, 2],
            0,
        ]
        return Translation(
            translation=translation,
            name="Scipion stored translation. Shifts in pixels.",
            input="Tilt-image",
            output="Tilt-image",
        )

    @staticmethod
    def _gen_rotation_transform(
        transformation_matrix: np.ndarray,
    ) -> CoordinateTransformation:
        row1: Vector3D = transformation_matrix[0, :].tolist()
        row1[-1] = 0
        row2: Vector3D = transformation_matrix[1, :].tolist()
        row2[-1] = 0
        row3: Vector3D = [0, 0, 1]
        affine_matrix: Matrix3x3 = [row1, row2, row3]
        return Affine(
            affine=affine_matrix,
            name="Scipion stored rotation",
            input="Tilt-image",
            output="Tilt-image",
        )


# scratch_dir = "/home/jjimenez/CZII/cets_scratch_dir"
# f_path = Path(
#     "/home/jjimenez/ScipionUserData/projects/chlamy/Runs/002094_ProtAreTomoAlignRecon/"
# )
# db_fn = f_path / "tiltseries_3.sqlite"
# sci_tsm = ScipionTiltSeries(db_fn)
# sci_tsm.scipion_to_cets(out_directory=scratch_dir)

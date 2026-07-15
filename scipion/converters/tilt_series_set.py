import ast
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from cets_data_model.models.models import (
    TiltImage,
    Axis,
    AxisType,
    CoordinateSystem,
    CoordinateTransformation,
    Translation,
    Vector3D,
    Affine,
    Matrix3x3,
    TiltSeries,
    CTFMetadata,
    ProjectionAlignment,
    Alignment,
)
from cets_data_model.utils.image_utils import get_mrc_info
from scipion.constants import (
    TS_ID,
    TILT_SERIES_FIELDS,
    FILE_NAME,
    INDEX,
    TILT_ANGLE,
    ACCUMULATED_DOSE,
    # ACQUISITION_ORDER,
    TRANSFORMATION_MATRIX,
    ODD_EVEN_FN,
    CTF_CORRECTED,
    CLASSES_TBL,
    OBJECTS_TBL,
)
from scipion.converters.base_converter import BaseConverter
from scipion.utils.utils import write_ts_set_yaml, write_alignment_set_yaml
from scipion.utils.utils_sqlite import (
    connect_db,
    map_classes_table,
    get_from_obj_tbl,
    get_row_value,
)


class ScipionSetOfTiltSeries(BaseConverter):
    def __init__(self, sqlite_path: os.PathLike):
        super().__init__(sqlite_path)
        self.img_x = -1
        self.img_y = -1

    def scipion_to_cets(
        self,
        ctf_md: Dict[str, List[CTFMetadata]] | None = None,
        out_directory: str | None = None,
    ) -> Tuple[List[TiltSeries], List[Alignment]] | None:
        """Converts a set of tilt-series from Scipion into CETS metadata.

        In the current data model the per-projection alignment is NOT stored inside each
        tilt-image's ``coordinate_transformations``. Instead it is represented with the
        dedicated ``ProjectionAlignment`` structure (one per tilt-image, holding the
        ``[Translation, Affine]`` pair in its ``sequence``), and all of them are aggregated
        into a single ``Alignment`` per tilt-series. ``Alignment`` objects live under
        ``Region.alignments``; since this converter emits bare ``TiltSeries`` (not a
        ``Region``), the alignments are returned alongside them for higher-level assembly.
        The returned lists are index-aligned: ``alignments[i]`` is the alignment of
        ``tilt_series[i]``, and its i-th ``ProjectionAlignment`` corresponds to that
        tilt-series' i-th ``images`` entry (positional binding).

        :param ctf_md: dictionary of type key: tilt-series id, value: list of CTF Metadata
        containing the CTFMetadata of corresponding to all the tilt-images that compose the tilt-series
        of id equal to the key of the dictionary. It can be obtained using the method
        ScipionCtfSeries.scipion_to_cets.
        :type ctf_md: Dict[str, List[CTFMetadata]] or None, optional, Defaults to None

        :param out_directory: name of the directory in which the tilt-series
        .yaml files (one per tilt-series) will be written.
        :type out_directory: pathlib.Path or str, optional, Defaults to None
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
                ti_sql_fields = self._get_sql_fields(ts_class_dict, TILT_SERIES_FIELDS)

                # Coordinate system
                axis_z = Axis(
                    name="Z",
                    axis_unit="pixel",
                    axis_type=AxisType.space,
                )
                coordinate_systems = CoordinateSystem(name="SCIPION", axes=[axis_z])

                cursor = conn.cursor()
                tilt_series_list = []
                alignments_list = []
                for i, ts_id in enumerate(ts_ids):
                    print(f"tsId = {ts_id}. Loading the tilt-series...")
                    # Manage the CTFMetadata
                    ctf_md_list = ctf_md.get(ts_id, None) if ctf_md else None
                    # Read the tilt-images table
                    ti_list = []
                    projection_alignments = []
                    tilt_images_table_name = self._get_ts_obj_tbl_name(ts_id)
                    query = f'SELECT {ti_sql_fields} FROM "{tilt_images_table_name}"'
                    cursor.execute(query)  # execute the query
                    for row in cursor.fetchall():
                        ti, projection_alignment, odd_fn, even_fn = (
                            self._ti_from_sqlite_row(
                                row, ts_class_dict, coordinate_systems
                            )
                        )
                        self._add_ctf_md(ti, i, ctf_md_list)
                        ti_list.append(ti)
                        # One ProjectionAlignment per projection (index-aligned with ti_list).
                        projection_alignments.append(projection_alignment)

                    # Tilt-series
                    ts = TiltSeries(
                        id=ts_id,  # TODO: define this
                        movie_stack_series_id=ts_id,  # TODO: define this
                        path=ti_list[-1].path,
                        even_path=even_fn,
                        odd_path=odd_fn,
                        ctf_corrected=bool(ctf_corrected_list[i]),
                        images=ti_list,
                    )
                    tilt_series_list.append(ts)
                    # Alignment for this tilt-series (meant to be placed under Region.alignments),
                    # linked back to the tilt-series via tilt_series_id.
                    alignments_list.append(
                        Alignment(
                            tilt_series_id=ts_id,
                            projection_alignments=projection_alignments,
                        )
                    )

                if out_directory:
                    write_ts_set_yaml(tilt_series_list, Path(out_directory))
                    write_alignment_set_yaml(
                        tilt_series_list, alignments_list, Path(out_directory)
                    )
                return tilt_series_list, alignments_list
        return None

    def _ti_from_sqlite_row(
        self,
        row: sqlite3.Row,
        ts_class_dict: Dict[str, str],
        coord_system: CoordinateSystem,
    ) -> Tuple[TiltImage, ProjectionAlignment, Optional[str], Optional[str]]:
        # Read image info
        ts_file = get_row_value(row, ts_class_dict, FILE_NAME)
        ts_fn = self.scipion_prj_path / ts_file if ts_file else self.scipion_prj_path
        if self.img_x < 0:
            img_info = get_mrc_info(ts_fn)
            self.img_x = img_info.size_x
            self.img_y = img_info.size_y
        # Get the odd / even filenames
        even_fn, odd_fn = None, None
        odd_even_fn = get_row_value(row, ts_class_dict, ODD_EVEN_FN)
        if odd_even_fn:
            even_fn, odd_fn = sorted(odd_even_fn.split(","))
        # Get the transformation matrix
        tr_matrix_str = get_row_value(row, ts_class_dict, TRANSFORMATION_MATRIX)
        tr_matrix = np.array(ast.literal_eval(tr_matrix_str))
        ts_id = get_row_value(row, ts_class_dict, TS_ID)
        section = get_row_value(row, ts_class_dict, INDEX)
        # Unique tilt-image id within the tilt-series (derived from the ts id + section).
        tilt_image_id = f"{ts_id}_{section}"

        # Create the tilt-image
        ti = TiltImage(
            id=tilt_image_id,
            movie_stack_id=ts_id,  # TODO: define this
            path=str(ts_fn),
            # even_path=even_fn,
            # odd_path=odd_fn,
            # acquisition_order=get_row_value(row, ts_class_dict, ACQUISITION_ORDER),
            section=section,
            nominal_tilt_angle=get_row_value(row, ts_class_dict, TILT_ANGLE),
            accumulated_dose=get_row_value(row, ts_class_dict, ACCUMULATED_DOSE),
            width=self.img_x,
            height=self.img_y,
            coordinate_systems=[coord_system],
            # Alignment is no longer stored here; it lives in the ProjectionAlignment below.
        )
        # ProjectionAlignment linked to its tilt-image by tilt_image_id.
        projection_alignment = self._gen_projection_alignment(
            tr_matrix,
            projection_alignment_id=f"{ts_id}_align_{section}",
            tilt_image_id=tilt_image_id,
        )
        return ti, projection_alignment, odd_fn, even_fn

    @staticmethod
    def _get_ts_classes_tbl_name(ts_id: str) -> str:
        return f"{ts_id}_{CLASSES_TBL}"

    @staticmethod
    def _get_ts_obj_tbl_name(ts_id: str) -> str:
        return f"{ts_id}_{OBJECTS_TBL}"

    def _gen_projection_alignment(
        self,
        transformation_matrix: np.ndarray,
        projection_alignment_id: str = "",
        tilt_image_id: str | None = None,
    ) -> ProjectionAlignment:
        """Wraps the per-projection translation and affine rotation into a
        ProjectionAlignment (order preserved: translation first, affine second).

        :param projection_alignment_id: unique id for this ProjectionAlignment.
        :param tilt_image_id: id of the TiltImage this alignment applies to.
        """
        return ProjectionAlignment(
            id=projection_alignment_id,
            tilt_image_id=tilt_image_id,
            sequence=[
                self._gen_translation_transform(transformation_matrix),
                self._gen_rotation_transform(transformation_matrix),
            ],
            name="Scipion projection alignment.",
            input="Tilt-image",
            output="Aligned tilt-image",
        )

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

    @staticmethod
    def _add_ctf_md(ti: TiltImage, index: int, ctf_md: List[CTFMetadata] | None = None):
        ti.ctf_metadata = ctf_md[index] if ctf_md else None

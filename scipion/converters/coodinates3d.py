from cets_data_model.models.models import (
    PointSet3D,
    AnnotationType,
    CoordinateSystem,
    Axis,
    AxisType,
)
from scipion.constants import (
    COORD_3D_FIELDS,
    OBJECTS_TBL,
    TOMO_ID,
    COORD_X,
    COORD_Y,
    COORD_Z,
)
from scipion.converters.base_converter import BaseConverter
from scipion.utils.utils_sqlite import connect_db, map_classes_table, get_row_value


coordinates_system = [
    CoordinateSystem(
        name="Scipion",
        axes=[Axis(name="ZYZ", axis_type=AxisType.space, axis_unit="pixel")],
    )
]


class ScipionSetOfCoordinates3D(BaseConverter):
    def scipion_to_cets(
        self,
        tomo_id: str,
        # out_directory: str | None = None
    ) -> PointSet3D | None:
        """Converts the set of coordinates corresponding to the introduced tomogram identifier
        into CETS metadata.

        Picked coordinates are represented as a ``PointSet3D``
        annotation. The ``PointSet3D`` holds the coordinates in ``origin3D``
        (an Nx3 array) and links back to the tomogram they were picked in through
        ``source_tomogram_id``. This annotation is meant to be stored under a
        ``Region.annotations`` list.

        Scipion stores the coordinates of every tomogram together, so this method returns
        one ``PointSet3D`` per tomogram (matching the "one PointSet3D per tomogram" decision
        baked into the model, where ``source_tomogram_id`` sits on the annotation).

        :param tomo_id: Scipion tomogram identifier. It is used to indicate the tomogram from which the
        coordinates will be converted, as in Scipion the coordinates from all the tomograms are
        stored together.
        :type tomo_id: str.
        """
        db_connection = connect_db(self.db_path)
        if db_connection is not None:
            with db_connection as conn:
                # Map the table Classes and get some values from the table Objects
                coord_set_class_dict = map_classes_table(conn)

                # Sqlite fields of the data to be read from each tomogram
                coord_sql_fields = self._get_sql_fields(
                    coord_set_class_dict, COORD_3D_FIELDS
                )

                cursor = conn.cursor()
                tomo_id_col_name = coord_set_class_dict[TOMO_ID]
                query = f'SELECT {coord_sql_fields} FROM "{OBJECTS_TBL}" WHERE {tomo_id_col_name}="{tomo_id}"'
                cursor.execute(query)  # execute the query
                origin_3d = []
                for row in cursor:
                    # TODO (open question #1): the per-coordinate Euler orientation
                    # (EULER_MATRIX) cannot be stored on a PointSet3D, which only carries
                    # positions (origin3D) plus set-level transforms. If per-point
                    # orientations must be preserved for picked coordinates, use
                    # PointVectorSet3D / PointMatrixSet3D instead. For now only the
                    # positions are converted.
                    origin_3d.append(
                        [
                            get_row_value(row, coord_set_class_dict, COORD_X),
                            get_row_value(row, coord_set_class_dict, COORD_Y),
                            get_row_value(row, coord_set_class_dict, COORD_Z),
                        ]
                    )
                if not origin_3d:
                    return None
                point_set = PointSet3D(
                    # TODO (open question #3): id-generation policy. The tomogram id is
                    # used here so that AnnotationReference.source_annotation_id can resolve
                    # this annotation. It must be unique within its Region.annotations.
                    id=f"scipion_coords_{tomo_id}",
                    name=f"Scipion coordinates for {tomo_id}",
                    annotation_type=AnnotationType.point_set_3D,
                    source_tomogram_id=tomo_id,
                    origin3D=origin_3d,
                    coordinate_systems=coordinates_system,
                )
                # if out_directory:
                #     write_coords_set_yaml(point_set, tomo_id, Path(out_directory))
                return point_set
        return None

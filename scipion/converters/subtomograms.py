import ast
from cets_data_model.models.models import (
    Particle3DSet,
    CoordinateSystem,
    Axis,
    AxisType,
    AxisUnit,
    SpaceAxis,
    Particle3D,
)
from cets_data_model.utils.image_utils import get_mrc_info
from scipion.constants import (
    OBJECTS_TBL,
    SUBTOMO_FIELDS,
    SUBTOMO_ID,
    FILE_NAME,
    SUBTOMO_X,
    SUBTOMO_Y,
    SUBTOMO_Z,
    SUBTOMO_COORD_MATRIX,
    SUBTOMO_TRANSFORM_MATRIX,
)
from scipion.converters.base_converter import BaseConverter
from scipion.utils.utils_sqlite import connect_db, map_classes_table, get_row_value


coordinates_system = [
    CoordinateSystem(
        name="Scipion",
        axes=[
            Axis(name=SpaceAxis.ZYZ, axis_type=AxisType.space, axis_unit=AxisUnit.pixel)
        ],
    )
]


class ScipionSetOfSubtomogras(BaseConverter):
    def scipion_to_cets(
        self,
        tomo_id: str,
        # out_directory: str | None = None
    ) -> Particle3DSet | None:
        """Converts the set of subtomograms corresponding to the introduced tomogram identifier
        into CETS metadata.

        :param tomo_id: tomogram identifier. It is used to indicate the tomogram from which the
        subtomograms will be converted, as in Scipion the subtomograms from all the tomograms are
        stored together.
        """
        db_connection = connect_db(self.db_path)
        if db_connection is not None:
            with db_connection as conn:
                # Map the table Classes and get some values from the table Objects
                coord_set_class_dict = map_classes_table(conn)

                # Sqlite fields of the data to be read from each tomogram
                coord_sql_fields = self._get_sql_fields(
                    coord_set_class_dict, SUBTOMO_FIELDS
                )

                cursor = conn.cursor()
                tomo_id_col_name = coord_set_class_dict[SUBTOMO_ID]
                query = f'SELECT {coord_sql_fields} FROM "{OBJECTS_TBL}" WHERE {tomo_id_col_name}="{tomo_id}"'
                cursor.execute(query)  # execute the query
                coord_list = []
                for row in cursor:
                    subtomo_fn = get_row_value(row, coord_set_class_dict, FILE_NAME)
                    subtomo_fn = (
                        self.scipion_prj_path / subtomo_fn
                        if subtomo_fn
                        else self.scipion_prj_path
                    )
                    img_info = get_mrc_info(subtomo_fn)

                    euler_matrix = ast.literal_eval(
                        get_row_value(row, coord_set_class_dict, SUBTOMO_COORD_MATRIX)
                    )
                    coordinate_transform = self._gen_subvolume_transform(euler_matrix)
                    subtomo_euler_matrix = ast.literal_eval(
                        get_row_value(
                            row, coord_set_class_dict, SUBTOMO_TRANSFORM_MATRIX
                        )
                    )
                    subtomo_transform = self._gen_subvolume_transform(
                        subtomo_euler_matrix, is_coordinate=False
                    )

                    position = [
                        get_row_value(row, coord_set_class_dict, SUBTOMO_X),
                        get_row_value(row, coord_set_class_dict, SUBTOMO_Y),
                        get_row_value(row, coord_set_class_dict, SUBTOMO_Z),
                    ]
                    print(position)
                    coordinate3d = Particle3D(
                        path=str(subtomo_fn),
                        width=img_info.size_x,
                        height=img_info.size_y,
                        depth=img_info.size_z,
                        position=position,
                        coordinate_transformations=[
                            coordinate_transform,
                            subtomo_transform,
                        ],
                    )
                    coord_list.append(coordinate3d)
                coordinates = Particle3DSet(
                    particles=coord_list,
                    coordinate_systems=coordinates_system,
                )
                # if out_directory:
                #     write_coords_set_yaml(coordinates, Path(out_directory))
                return coordinates
        return None

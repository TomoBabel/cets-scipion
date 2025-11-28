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
from scipion.constants import (
    COORD_3D_FIELDS,
    OBJECTS_TBL,
    TOMO_ID,
    COORD_X,
    COORD_Y,
    COORD_Z,
    EULER_MATRIX,
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


class ScipionSetOfCoordinates3D(BaseConverter):
    def scipion_to_cets(
        self,
        tomo_id: str,
        # out_directory: str | None = None
    ) -> Particle3DSet | None:
        """Converts the set of coordinates corresponding to the introduced tomogram identifier
        into CETS metadata.

        :param tomo_id: tomogram identifier. It is used to indicate the tomogram from which the
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
                coord_list = []
                for row in cursor:
                    euler_matrix = ast.literal_eval(
                        get_row_value(row, coord_set_class_dict, EULER_MATRIX)
                    )
                    coordinate_transform = self._gen_subvolume_transform(euler_matrix)

                    coordinate3d = Particle3D(
                        position=[
                            get_row_value(row, coord_set_class_dict, COORD_X),
                            get_row_value(row, coord_set_class_dict, COORD_Y),
                            get_row_value(row, coord_set_class_dict, COORD_Z),
                        ],
                        coordinate_transformations=[coordinate_transform],
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

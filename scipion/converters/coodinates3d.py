from cets_data_model.models.models import CoordinateSet3D
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


class ScipionSetOfCoordinates3D(BaseConverter):
    def scipion_to_cets(
        self, ts_id: str, out_directory: str | None = None
    ) -> CoordinateSet3D | None:
        """Converts the set of coordinates corresponding to the introduced tomogram identifier
        into CETS metadata.

        :param ts_id: tomogram identifier. It is used to indicate the tomogram from which the
        coordinates will be converted, as in Scipion the coordinates from all the tomorgams are
        stored together.

        :param out_directory: name of the directory in which the tilt-series
        .yaml files (one per tilt-series) will be written.
        :type out_directory: pathlib.Path or str, optional, Defaults to None
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
                query = f'SELECT {coord_sql_fields} FROM "{OBJECTS_TBL}" WHERE {TOMO_ID}="{ts_id}"'
                cursor.execute(query)  # execute the query
                coord_list = []
                for row in cursor:
                    coord_list.append(
                        [
                            get_row_value(row, coord_set_class_dict, COORD_X),
                            get_row_value(row, coord_set_class_dict, COORD_Y),
                            get_row_value(row, coord_set_class_dict, COORD_Z),
                        ]
                    )
                return CoordinateSet3D(
                    coordinates=coord_list,
                    origin3D=None,  # TODO: fill this
                    coordinate_systems=None,  # TODO: fill this
                    coordinate_transformations=None,  # TODO: fill this
                )
        return None

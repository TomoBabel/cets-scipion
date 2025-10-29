from cets_data_model.models.models import Tomogram
from cets_data_model.utils.image_utils import get_mrc_info
from scipion.constants import (
    TOMOGRAM_FIELDS,
    OBJECTS_TBL,
    FILE_NAME,
    TS_ID,
    ODD_EVEN_TOMOS_FN,
    CTF_CORRECTED,
)
from scipion.converters.base_converter import BaseConverter
from scipion.utils.utils_sqlite import connect_db, map_classes_table, get_row_value


class ScipionSetOfTomograms(BaseConverter):
    def scipion_to_cets(self):
        """Converts a set of tomograms from Scipion into CETS metadata."""
        db_connection = connect_db(self.db_path)
        if db_connection is not None:
            with db_connection as conn:
                # Map the table Classes and get some values from the table Objects
                tomo_set_class_dict = map_classes_table(conn)

                # Sqlite fields of the data to be read from each tomogram
                tomo_sql_fields = self._get_sql_fields(
                    tomo_set_class_dict, TOMOGRAM_FIELDS
                )

                cursor = conn.cursor()
                query = f'SELECT {tomo_sql_fields} FROM "{OBJECTS_TBL}"'
                cursor.execute(query)  # execute the query
                tomo_list = []
                for row in cursor:
                    # Read tomogram info
                    tomo_file = get_row_value(row, tomo_set_class_dict, FILE_NAME)
                    tomo_fn = (
                        self.scipion_prj_path / tomo_file
                        if tomo_file
                        else self.scipion_prj_path
                    )
                    img_info = get_mrc_info(tomo_fn)
                    # Get the odd / even filenames
                    even_fn, odd_fn = None, None
                    odd_even_fn = get_row_value(
                        row, tomo_set_class_dict, ODD_EVEN_TOMOS_FN
                    )
                    if odd_even_fn:
                        even_fn, odd_fn = sorted(odd_even_fn.split(","))
                    tomo = Tomogram(
                        tomo_id=get_row_value(row, tomo_set_class_dict, TS_ID),
                        path=str(tomo_fn),
                        even_path=even_fn,
                        odd_path=odd_fn,
                        width=img_info.size_x,
                        height=img_info.size_y,
                        depth=img_info.size_z,
                        coordinate_systems=None,  # TODO: what about this in tomograms?
                        coordinate_transformations=None,
                        ctf_corrected=get_row_value(
                            row, tomo_set_class_dict, CTF_CORRECTED
                        ),
                    )
                    tomo_list.append(tomo)

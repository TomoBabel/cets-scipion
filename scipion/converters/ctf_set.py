import sqlite3
from typing import List, Dict

from cets_data_model.models.models import CTFMetadata
from scipion.constants import (
    TS_ID,
    CTF_TOMO_SERIES_FIELDS,
    CLASSES_TBL,
    OBJECTS_TBL,
    DEFOCUS_U,
    DEFOCUS_V,
    DEFOCUS_ANGLE,
    ACQUISITION_ORDER,
    PHASE_SHIFT,
)
from scipion.converters.base_converter import BaseConverter
from scipion.utils.utils_sqlite import (
    connect_db,
    map_classes_table,
    get_from_obj_tbl,
    get_row_value,
)


class ScipionSetOfCtf(BaseConverter):
    def scipion_to_cets(
        self,
    ) -> Dict[str, List[CTFMetadata]] | None:
        """Converts a set of CTF from Scipion into CETS metadata."""
        db_connection = connect_db(self.db_path)
        if db_connection is not None:
            with db_connection as conn:
                # Map the table Classes and get some values from the table Objects
                ctf_set_class_dict = map_classes_table(conn)
                ts_ids = get_from_obj_tbl(conn, TS_ID, ctf_set_class_dict)

                # Map the table Classes of the first CTFTomoSeries
                ctf_tomo_class_dict = map_classes_table(
                    conn, self._get_ctf_classes_tbl_name(1)
                )

                # Sqlite fields of the data to be read from each CTFTomoSeries
                ctf_sql_fields = self._get_sql_fields(
                    ctf_tomo_class_dict, CTF_TOMO_SERIES_FIELDS
                )

                cursor = conn.cursor()
                ctf_tomo_dict = {}
                for i, ts_id in enumerate(ts_ids):
                    print(f"tsId = {ts_id}. Loading the CTF series...")
                    # Read the CTF table
                    ctf_list = []
                    ctf_list_table_name = self._get_ctf_obj_tbl_name(i + 1)
                    query = f'SELECT {ctf_sql_fields} FROM "{ctf_list_table_name}"'
                    cursor.execute(query)  # execute the query
                    for row in cursor.fetchall():
                        print(dict(row))
                        ctf = self._ctf_from_sqlite_row(row, ctf_tomo_class_dict)
                        ctf_list.append(ctf)
                    ctf_tomo_dict[ts_id] = ctf_list
                return ctf_tomo_dict
        return None

    @staticmethod
    def _get_ctf_classes_tbl_name(ctf_set_row_index: int) -> str:
        """Returns the classes table name for each CTFTomoSeries (they are
        named id1_Classes, id2_classes, etc.).
        :param ctf_set_row_index: CTFTomoSeries index, from 1.
        """
        return f"id{ctf_set_row_index}_{CLASSES_TBL}"

    @staticmethod
    def _get_ctf_obj_tbl_name(ctf_set_row_index: int) -> str:
        """Returns the classes table name for each CTFTomoSeries (they are
        named id1_Classes, id2_classes, etc.).
        :param ctf_set_row_index: CTFTomoSeries index, from 1.
        """
        return f"id{ctf_set_row_index}_{OBJECTS_TBL}"

    @staticmethod
    def _ctf_from_sqlite_row(
        row: sqlite3.Row,
        ctf_tomo_class_dict: Dict[str, str],
    ) -> CTFMetadata:
        return CTFMetadata(
            defocus_u=get_row_value(row, ctf_tomo_class_dict, DEFOCUS_U),
            defocus_v=get_row_value(row, ctf_tomo_class_dict, DEFOCUS_V),
            defocus_angle=get_row_value(row, ctf_tomo_class_dict, DEFOCUS_ANGLE),
            phase_shift=get_row_value(row, ctf_tomo_class_dict, PHASE_SHIFT),
            acquisition_order=get_row_value(
                row, ctf_tomo_class_dict, ACQUISITION_ORDER
            ),
        )

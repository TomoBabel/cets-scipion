import sqlite3
from pathlib import Path

from scipion.constants import TS_ID, TILT_SERIES_FIELDS
from scipion.utils.utils import connect_db, map_classes_table, map_properties_table, get_tsm_ts_ids


class ScipionTiltSeries:

    def scipion_to_cets(self, db_path: Path):
        with connect_db(db_path) as conn:
            tsm_class_dict = map_classes_table(conn)
            prop_dict = map_properties_table(conn)
            ts_ids = get_tsm_ts_ids(conn, tsm_class_dict[TS_ID])
            ts_class_dict = map_classes_table(conn, self._get_ts_table_name(ts_ids[0]))

            # ti_list = []
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            sql_fields = ", ".join([f'"{ts_class_dict[field]}"' for field in TILT_SERIES_FIELDS])
            for ts_id in ts_ids:
                print(f"Loading tsId = {ts_id}")
                ts_table_name = self._get_ts_table_name(ts_id)
                # query = f'SELECT {sql_fields} FROM "{ts_table_name}"'
                query = f'SELECT * FROM "{ts_table_name}"'
                cursor.execute(query)  # execute the query
                # for row in cursor.fetchall():
                #     for field in TILT_SERIES_FIELDS:
                #         s_field = f'"{ts_class_dict[field]}"'
                #         print(f'{field}: {row[s_field]}')

                for row in cursor.fetchall():
                    print(dict(row))

    @staticmethod
    def _get_ts_table_name(ts_id: str) -> str:
        return ts_id + "_Classes"


f_path = Path("/home/jjimenez/ScipionUserData/projects/chlamy/Runs/002094_ProtAreTomoAlignRecon")
db_fn = f_path / 'tiltseries.sqlite'
sci_tsm = ScipionTiltSeries()
sci_tsm.scipion_to_cets(db_fn)


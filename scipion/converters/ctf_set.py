import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Optional

from cets_data_model.models.models import CTFMetadata
from scipion.constants import (
    TS_ID,
    CTF_TOMO_SERIES_FIELDS,
    CLASSES_TBL,
    OBJECTS_TBL,
    OBJ_ID,
    DEFOCUS_U,
    DEFOCUS_V,
    DEFOCUS_ANGLE,
    PHASE_SHIFT,
    INDEX,
    ACQUISITION_ORDER,
)
from scipion.converters.base_converter import BaseConverter
from scipion.utils.utils_sqlite import (
    connect_db,
    map_classes_table,
    get_row_value,
)


@dataclass
class ScipionCtfTomo:
    """CTF of a tilt-image of a Scipion CTFTomoSeries, together with the keys Scipion uses
    to match it with its tilt-image (``CTFTomoSeries.getCtfTomoFromTi``): the acquisition
    order (``_acqOrder``) and, if it is not stored, the index (``_index``, 1-based)."""

    ctf: CTFMetadata
    acq_order: Optional[int] = None
    index: Optional[int] = None


class ScipionSetOfCtf(BaseConverter):
    def scipion_to_cets(
        self,
    ) -> Dict[str, List[ScipionCtfTomo]] | None:
        """Converts a set of CTF from Scipion into CETS metadata.

        :returns: dictionary of type key: tilt-series id, value: list of the CTFs of the
        tilt-images of that tilt-series, each one with the acquisition order and index needed
        to match it with its tilt-image (see ``ScipionSetOfTiltSeries.scipion_to_cets``).
        """
        db_connection = connect_db(self.db_path)
        if db_connection is not None:
            with db_connection as conn:
                # Map the table Classes and get the object id and tsId of each CTFTomoSeries
                ctf_set_class_dict = map_classes_table(conn)
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT "{OBJ_ID}", "{ctf_set_class_dict[TS_ID]}" '
                    f'FROM "{OBJECTS_TBL}"'
                )
                ctf_series = [(row[0], row[1]) for row in cursor.fetchall()]

                ctf_tomo_dict = {}
                for obj_id, ts_id in ctf_series:
                    print(f"tsId = {ts_id}. Loading the CTF series...")
                    # The CTFs of each CTFTomoSeries are stored in the tables named after its
                    # object id (id<objId>_Classes / id<objId>_Objects), not after its
                    # position in the set.
                    ctf_tomo_class_dict = map_classes_table(
                        conn, self._get_ctf_classes_tbl_name(obj_id)
                    )
                    ctf_sql_fields = self._get_sql_fields(
                        ctf_tomo_class_dict, CTF_TOMO_SERIES_FIELDS
                    )
                    query = (
                        f"SELECT {ctf_sql_fields} "
                        f'FROM "{self._get_ctf_obj_tbl_name(obj_id)}"'
                    )
                    cursor.execute(query)  # execute the query
                    ctf_tomo_dict[ts_id] = [
                        self._ctf_from_sqlite_row(row, ctf_tomo_class_dict)
                        for row in cursor.fetchall()
                    ]
                return ctf_tomo_dict
        return None

    @staticmethod
    def _get_ctf_classes_tbl_name(ctf_series_obj_id: int) -> str:
        """Returns the classes table name of a CTFTomoSeries (id<objId>_Classes).
        :param ctf_series_obj_id: object id of the CTFTomoSeries in the set (column id).
        """
        return f"id{ctf_series_obj_id}_{CLASSES_TBL}"

    @staticmethod
    def _get_ctf_obj_tbl_name(ctf_series_obj_id: int) -> str:
        """Returns the objects table name of a CTFTomoSeries (id<objId>_Objects).
        :param ctf_series_obj_id: object id of the CTFTomoSeries in the set (column id).
        """
        return f"id{ctf_series_obj_id}_{OBJECTS_TBL}"

    @staticmethod
    def _ctf_from_sqlite_row(
        row: sqlite3.Row,
        ctf_tomo_class_dict: Dict[str, str],
    ) -> ScipionCtfTomo:
        acq_order = get_row_value(row, ctf_tomo_class_dict, ACQUISITION_ORDER)
        index = get_row_value(row, ctf_tomo_class_dict, INDEX)
        return ScipionCtfTomo(
            ctf=CTFMetadata(
                defocus_u=get_row_value(row, ctf_tomo_class_dict, DEFOCUS_U),
                defocus_v=get_row_value(row, ctf_tomo_class_dict, DEFOCUS_V),
                defocus_angle=get_row_value(row, ctf_tomo_class_dict, DEFOCUS_ANGLE),
                # Degrees; None if the CTF estimation did not fit it
                phase_shift=get_row_value(row, ctf_tomo_class_dict, PHASE_SHIFT),
            ),
            acq_order=int(acq_order) if acq_order is not None else None,
            index=int(index) if index is not None else None,
        )

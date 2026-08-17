import os
from pathlib import Path
from typing import List, Optional

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
from scipion.utils.utils import write_tomo_set_yaml
from scipion.utils.utils_sqlite import connect_db, map_classes_table, get_row_value


class ScipionSetOfTomograms(BaseConverter):
    def scipion_to_cets(
        self,
        out_directory: Optional[os.PathLike] = None,
    ) -> List[Tomogram] | None:
        """Converts a set of tomograms from Scipion into CETS metadata.

        Coordinates are now ``PointSet3D`` annotations (see ``ScipionSetOfCoordinates3D``) that
        live under ``Region.annotations``, and extracted subvolumes are ``ParticleMap`` objects
        under ``Average.particle_maps`` (see ``ScipionSetOfSubtomogras``). This converter is
        therefore tomogram-only; run the coordinate / subtomogram converters separately and
        assemble the ``Region`` / ``Dataset`` at a higher level.

        :param out_directory: name of the directory in which the tomogram
        .yaml files (one per tomogram) will be written.
        :type out_directory: os.PathLike, optional. Defaults to None.
        """
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
                    tomo_id = get_row_value(row, tomo_set_class_dict, TS_ID)
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
                        # TODO (open question #3): id-generation policy. The Scipion tomogram
                        # id (tsId) is reused as the Tomogram id; it must be unique within its
                        # Region. source_tomogram_id on the coordinate PointSet3D must match it.
                        id=tomo_id,
                        path=str(tomo_fn),
                        tilt_series_id=tomo_id,
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
                if out_directory:
                    write_tomo_set_yaml(tomo_list, Path(out_directory))
                return tomo_list
        return None

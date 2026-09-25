import ast
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from cets_data_model.models.models import (
    TiltImage,
    Translation,
    Affine,
    Matrix3x3,
    TiltSeries,
    CTFMetadata,
    ProjectionAlignment,
    Alignment,
    Instrument,
    AcquisitionSession,
)
from cets_data_model.utils.image_utils import get_mrc_info
from scipion.constants import (
    TS_ID,
    TILT_SERIES_FIELDS,
    FILE_NAME,
    INDEX,
    TILT_ANGLE,
    ACCUMULATED_DOSE,
    VOLTAGE,
    SPHERICAL_ABERRATION,
    AMPLITUDE_CONTRAST,
    # ACQUISITION_ORDER,
    TRANSFORMATION_MATRIX,
    ODD_EVEN_FN,
    CTF_CORRECTED,
    CLASSES_TBL,
    OBJECTS_TBL,
)
from scipion.converters.base_converter import BaseConverter
from scipion.utils.utils import (
    write_ts_set_yaml,
    write_alignment_set_yaml,
    write_instruments_yaml,
)
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
        # Merged (tilt-series) pixel size in Å; cached from the first stack read and used
        # for the array_to_physical scale and to express alignment shifts in Å.
        self.pixel_size = -1.0

    def scipion_to_cets(
        self,
        ctf_md: Dict[str, List[CTFMetadata]] | None = None,
        out_directory: str | None = None,
    ) -> (
        Tuple[
            List[TiltSeries],
            List[Alignment],
            List[Instrument],
            List[AcquisitionSession],
        ]
        | None
    ):
        """Converts a set of tilt-series from Scipion into CETS metadata.

        In the current data model the per-projection alignment is NOT stored inside each
        tilt-image's ``coordinate_transformations``. Instead it is represented with the
        dedicated ``ProjectionAlignment`` structure (one per tilt-image, holding the
        ``[Affine, Translation]`` pair in its ``sequence`` — rotation then shift, shifts in Å —
        and mapping the tilt-image physical frame to the tilt-series' shared physical frame),
        and all of them are aggregated
        into a single ``Alignment`` per tilt-series. ``Alignment`` objects live under
        ``Region.alignments``; since this converter emits bare ``TiltSeries`` (not a
        ``Region``), the alignments are returned alongside them for higher-level assembly.
        The returned lists are index-aligned: ``alignments[i]`` is the alignment of
        ``tilt_series[i]``, and its i-th ``ProjectionAlignment`` corresponds to that
        tilt-series' i-th ``images`` entry (positional binding).

        Microscope/session acquisition metadata is no longer stored on the tilt-images: it
        is emitted as a single dataset-level ``Instrument`` (voltage, spherical aberration)
        and a single ``AcquisitionSession`` (amplitude contrast, linked to the instrument via
        ``instrument_id``). Every tilt-series references the session via
        ``acquisition_session_id``. Both are returned as the 3rd and 4th elements for
        higher-level assembly onto ``Dataset.instruments`` / ``Dataset.acquisition_sessions``.

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

                cursor = conn.cursor()
                tilt_series_list = []
                alignments_list = []
                # Dataset-level acquisition metadata (constant across the set); built once
                # from the first tilt-image row and referenced by every tilt-series.
                instrument: Optional[Instrument] = None
                acquisition_session: Optional[AcquisitionSession] = None
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
                        # Build the Instrument / AcquisitionSession once (values are the
                        # same for every image in the set).
                        if instrument is None:
                            instrument, acquisition_session = self._build_acquisition(
                                row, ts_class_dict
                            )
                        ti, projection_alignment, odd_fn, even_fn = (
                            self._ti_from_sqlite_row(row, ts_class_dict)
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
                        # Link the tilt-series to its acquisition session.
                        acquisition_session_id=(
                            acquisition_session.id if acquisition_session else None
                        ),
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

                # Dataset-level collections (one element each for a single Scipion set).
                instruments = [instrument] if instrument else []
                acquisition_sessions = (
                    [acquisition_session] if acquisition_session else []
                )
                if out_directory:
                    write_ts_set_yaml(tilt_series_list, Path(out_directory))
                    write_alignment_set_yaml(
                        tilt_series_list, alignments_list, Path(out_directory)
                    )
                    write_instruments_yaml(
                        instruments, acquisition_sessions, Path(out_directory)
                    )
                return (
                    tilt_series_list,
                    alignments_list,
                    instruments,
                    acquisition_sessions,
                )
        return None

    def _ti_from_sqlite_row(
        self,
        row: sqlite3.Row,
        ts_class_dict: Dict[str, str],
    ) -> Tuple[TiltImage, ProjectionAlignment, Optional[str], Optional[str]]:
        # Read image info
        ts_file = get_row_value(row, ts_class_dict, FILE_NAME)
        ts_fn = self.scipion_prj_path / ts_file if ts_file else self.scipion_prj_path
        if self.img_x < 0:
            img_info = get_mrc_info(ts_fn)
            self.img_x = img_info.size_x
            self.img_y = img_info.size_y
            # Merged (tilt-series) pixel size in Å; backs the array_to_physical scale.
            self.pixel_size = img_info.apix_x if img_info.apix_x else 1.0
        # Get the odd / even filenames
        even_fn, odd_fn = None, None
        odd_even_fn = get_row_value(row, ts_class_dict, ODD_EVEN_FN)
        if odd_even_fn:
            even_fn, odd_fn = sorted(odd_even_fn.split(","))
        # Get the transformation matrix
        tr_matrix_str = get_row_value(row, ts_class_dict, TRANSFORMATION_MATRIX)
        tr_matrix = np.array(ast.literal_eval(tr_matrix_str))
        ts_id = get_row_value(row, ts_class_dict, TS_ID)
        section = int(get_row_value(row, ts_class_dict, INDEX))
        # Unique tilt-image id within the tilt-series (derived from the ts id + section).
        tilt_image_id = f"{ts_id}_{section}"

        # Every image gets an array (pixel, unitless) and a physical (Å) coordinate system
        # plus exactly one canonical array_to_physical scale transformation (spec constraint).
        image_cs_name = f"tilt_image_{section:03d}"
        array_cs, physical_cs = self._gen_coordinate_systems(image_cs_name, ndim=2)
        array_to_physical = self._gen_array_to_physical(
            self.pixel_size, array_cs.name, physical_cs.name, ndim=2
        )

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
            # Microscope/session acquisition constants no longer live on the tilt-image;
            # they are emitted as Instrument / AcquisitionSession (see _build_acquisition).
            width=self.img_x,
            height=self.img_y,
            coordinate_systems=[array_cs, physical_cs],
            coordinate_transformations=[array_to_physical],
        )
        # ProjectionAlignment: maps this projection's physical frame to the tilt-series'
        # shared aligned physical frame; linked to the tilt-image by tilt_image_id.
        projection_alignment = self._gen_projection_alignment(
            tr_matrix,
            self.pixel_size,
            projection_alignment_id=f"{ts_id}_align_{section}",
            tilt_image_id=tilt_image_id,
            input_cs=physical_cs.name,
            output_cs=f"tilt_series_{ts_id}_physical",
            section=section,
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
        pixel_size: float,
        projection_alignment_id: str = "",
        tilt_image_id: str | None = None,
        input_cs: str | None = None,
        output_cs: str | None = None,
        section: int = 0,
    ) -> ProjectionAlignment:
        """Wraps the per-projection rotation and shift into a ProjectionAlignment.

        The ``sequence`` is ``[Affine, Translation]`` (rotation applied first, then shift),
        mapping the tilt-image physical frame (``input_cs``) to the tilt-series' shared
        aligned physical frame (``output_cs``). Shifts are converted from pixels to Å.

        :param projection_alignment_id: unique id for this ProjectionAlignment.
        :param tilt_image_id: id of the TiltImage this alignment applies to.
        :param input_cs: name of the tilt-image physical coordinate system.
        :param output_cs: name of the shared tilt-series aligned physical coordinate system.
        :param section: 0-based projection index, used for the alignment name.
        """
        return ProjectionAlignment(
            id=projection_alignment_id,
            tilt_image_id=tilt_image_id,
            sequence=[
                self._gen_rotation_transform(transformation_matrix),
                self._gen_translation_transform(transformation_matrix, pixel_size),
            ],
            name=f"alignment_tilt_{section:03d}",
            input=input_cs,
            output=output_cs,
        )

    @staticmethod
    def _gen_translation_transform(
        transformation_matrix: np.ndarray, pixel_size: float
    ) -> Translation:
        """Per-projection shift (x, y), converted from pixels to Å (physical frame)."""
        translation = [
            float(transformation_matrix[0, 2]) * pixel_size,
            float(transformation_matrix[1, 2]) * pixel_size,
        ]
        return Translation(translation=translation)

    @staticmethod
    def _gen_rotation_transform(
        transformation_matrix: np.ndarray,
    ) -> Affine:
        """Per-projection rotation as a homogeneous 3x3 affine (translation column zeroed);
        the rotation is dimensionless, so it is the same in the pixel and physical frames."""
        row1 = [float(v) for v in transformation_matrix[0, :].tolist()]
        row1[-1] = 0.0
        row2 = [float(v) for v in transformation_matrix[1, :].tolist()]
        row2[-1] = 0.0
        row3 = [0.0, 0.0, 1.0]
        affine_matrix: Matrix3x3 = [row1, row2, row3]
        return Affine(affine=affine_matrix)

    @staticmethod
    def _add_ctf_md(ti: TiltImage, index: int, ctf_md: List[CTFMetadata] | None = None):
        ti.ctf_metadata = ctf_md[index] if ctf_md else None

    @staticmethod
    def _build_acquisition(
        row: sqlite3.Row, ts_class_dict: Dict[str, str]
    ) -> Tuple[Instrument, AcquisitionSession]:
        """Builds the dataset-level Instrument and AcquisitionSession from a Scipion
        acquisition row.

        ``voltage`` is a physical instrument property; ``spherical_aberration`` and
        ``amplitude_contrast`` are kept at the session level. ``get_row_value`` returns None
        when a column is absent, so older sqlite files still work. ``dose_rate`` is
        intentionally left unset: Scipion stores dose-per-frame (e-/A^2), not a per-second
        rate (e-/A^2/s) as CETS ``dose_rate`` expects.
        """
        instrument = Instrument(
            id="instrument_0",
            voltage=get_row_value(row, ts_class_dict, VOLTAGE),
        )
        acquisition_session = AcquisitionSession(
            id="session_0",
            instrument_id=instrument.id,
            amplitude_contrast=get_row_value(row, ts_class_dict, AMPLITUDE_CONTRAST),
            spherical_aberration=get_row_value(
                row, ts_class_dict, SPHERICAL_ABERRATION
            ),
        )
        return instrument, acquisition_session

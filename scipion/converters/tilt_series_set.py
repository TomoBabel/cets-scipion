import ast
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Protocol, Sequence, TypeVar

import numpy as np

from cets_data_model.models.models import (
    TiltImage,
    Translation,
    Affine,
    Matrix3x3,
    TiltSeries,
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
    TILT_AXIS_ANGLE,
    ACQUISITION_ORDER,
    TRANSFORMATION_MATRIX,
    ODD_EVEN_FN,
    CTF_CORRECTED,
    CLASSES_TBL,
    OBJECTS_TBL,
)
from scipion.converters.base_converter import BaseConverter
from scipion.converters.ctf_set import ScipionCtfTomo
from scipion.utils.utils import (
    validate_file,
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


@dataclass
class ScipionTiltAngle:
    """Tilt angle of a tilt-image of a Scipion tilt-series, with the acquisition order and
    index (1-based) that identify the tilt-image."""

    angle: Optional[float]
    acq_order: Optional[int] = None
    index: Optional[int] = None


class _HasAcqOrderAndIndex(Protocol):
    acq_order: Optional[int]
    index: Optional[int]


_Matchable = TypeVar("_Matchable", bound=_HasAcqOrderAndIndex)


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
        ctf_md: Dict[str, List[ScipionCtfTomo]] | None = None,
        out_directory: str | None = None,
        nominal_ts_db_path: os.PathLike | None = None,
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

        :param ctf_md: dictionary of type key: tilt-series id, value: list of the CTFs of the
        tilt-images of that tilt-series, as returned by ScipionSetOfCtf.scipion_to_cets. Each
        CTF is assigned to its tilt-image as Scipion does (CTFTomoSeries.getCtfTomoFromTi): by
        acquisition order or, if it is not stored, by index. A tilt-image with no CTF keeps
        ctf_metadata empty.
        :type ctf_md: Dict[str, List[ScipionCtfTomo]] or None, optional, Defaults to None

        :param out_directory: name of the directory in which the tilt-series
        .yaml files (one per tilt-series) will be written.
        :type out_directory: pathlib.Path or str, optional, Defaults to None

        :param nominal_ts_db_path: sqlite file of the Scipion SetOfTiltSeries before the
        alignment (e.g. the imported one), to read the nominal tilt angles from.

        Scipion stores a single tilt angle per tilt-image (``_tiltAngle``), which the
        alignment protocols overwrite with the refined one. So, for the tilt-images that are
        aligned (they store a transformation matrix), it is stored as the
        ``refined_tilt_angle`` of their ``ProjectionAlignment``, and their
        ``nominal_tilt_angle`` is read from ``nominal_ts_db_path`` (matched by acquisition
        order or index), or left empty if it is not given. For the tilt-images that are not
        aligned, it is the ``nominal_tilt_angle``.
        :type nominal_ts_db_path: os.PathLike, optional, Defaults to None
        """
        nominal_angles = (
            self._load_nominal_tilt_angles(nominal_ts_db_path)
            if nominal_ts_db_path
            else None
        )
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
                    rows = cursor.fetchall()
                    nominal_list = (
                        nominal_angles.get(ts_id)
                        if nominal_angles is not None
                        else None
                    )
                    if nominal_angles is not None and nominal_list is None:
                        raise ValueError(
                            f"tsId = {ts_id}. It is not in the tilt-series set "
                            f"{nominal_ts_db_path}, given to read the nominal tilt angles."
                        )
                    for row in rows:
                        # Build the Instrument / AcquisitionSession once (values are the
                        # same for every image in the set).
                        if instrument is None:
                            instrument, acquisition_session = self._build_acquisition(
                                row, ts_class_dict
                            )
                        ti, projection_alignment, odd_fn, even_fn = (
                            self._ti_from_sqlite_row(row, ts_class_dict, nominal_list)
                        )
                        self._add_ctf_md(ti, row, ts_class_dict, ctf_md_list)
                        ti_list.append(ti)
                        # One ProjectionAlignment per projection (index-aligned with ti_list).
                        projection_alignments.append(projection_alignment)
                    if nominal_list is None and any(
                        ti.nominal_tilt_angle is None for ti in ti_list
                    ):
                        print(
                            f"tsId = {ts_id}. The tilt-series is aligned, so its tilt "
                            f"angles are stored as refined. Its nominal tilt angles are "
                            f"left empty: provide nominal_ts_db_path to read them."
                        )

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
        nominal_list: List[ScipionTiltAngle] | None = None,
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
        # Get the transformation matrix. Tilt-series that have not been aligned (e.g. just
        # imported) do not store it: their alignment is the identity (no rotation, no shift).
        tr_matrix_str = get_row_value(row, ts_class_dict, TRANSFORMATION_MATRIX)
        is_aligned = bool(tr_matrix_str) and tr_matrix_str != "None"
        tr_matrix = (
            np.array(ast.literal_eval(tr_matrix_str)) if is_aligned else np.eye(3)
        )
        ts_id = get_row_value(row, ts_class_dict, TS_ID)
        # Scipion's only tilt angle is the refined one once the tilt-series is aligned
        tilt_angle = get_row_value(row, ts_class_dict, TILT_ANGLE)
        if is_aligned:
            refined_tilt_angle = tilt_angle
            nominal_tilt_angle = None
            if nominal_list is not None:
                nominal = self._match_by_acq_order_or_index(
                    row, ts_class_dict, nominal_list, "nominal tilt angles"
                )
                if nominal is None:
                    raise ValueError(
                        f"tsId = {ts_id}. No nominal tilt angle matches the tilt-image "
                        f"of index {get_row_value(row, ts_class_dict, INDEX)}."
                    )
                nominal_tilt_angle = nominal.angle
        else:
            refined_tilt_angle = None
            nominal_tilt_angle = tilt_angle
        # Scipion's index of the image in the stack is 1-based; CETS sections are 0-based.
        section = int(get_row_value(row, ts_class_dict, INDEX)) - 1
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
            # Scipion records the temporal collection order per tilt image (_acqOrder);
            # map it straight to the CETS field instead of re-deriving it from the dose.
            acquisition_order=get_row_value(row, ts_class_dict, ACQUISITION_ORDER),
            section=section,
            nominal_tilt_angle=nominal_tilt_angle,
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
        projection_alignment.refined_tilt_angle = refined_tilt_angle
        return ti, projection_alignment, odd_fn, even_fn

    def _load_nominal_tilt_angles(
        self, ts_db_path: os.PathLike
    ) -> Dict[str, List[ScipionTiltAngle]]:
        """Tilt angles of each tilt-series of a Scipion SetOfTiltSeries that has not been
        aligned, with the acquisition order and index to match them with the tilt-images.

        :param ts_db_path: sqlite file of the SetOfTiltSeries.
        """
        ts_db = validate_file(ts_db_path, ".sqlite")
        conn = connect_db(ts_db)
        if conn is None:
            raise Exception(f"Unable to read the tilt-series from {ts_db}")
        fields = [TS_ID, INDEX, ACQUISITION_ORDER, TILT_ANGLE, TRANSFORMATION_MATRIX]
        angles: Dict[str, List[ScipionTiltAngle]] = {}
        try:
            ts_ids = get_from_obj_tbl(conn, TS_ID, map_classes_table(conn))
            cursor = conn.cursor()
            for ts_id in ts_ids:
                class_dict = map_classes_table(
                    conn, self._get_ts_classes_tbl_name(ts_id)
                )
                cursor.execute(
                    f"SELECT {self._get_sql_fields(class_dict, fields)} "
                    f'FROM "{self._get_ts_obj_tbl_name(ts_id)}"'
                )
                ts_angles = []
                for row in cursor.fetchall():
                    matrix = get_row_value(row, class_dict, TRANSFORMATION_MATRIX)
                    if matrix and matrix != "None":
                        raise ValueError(
                            f"tsId = {ts_id}. The tilt-series of {ts_db} are aligned, so "
                            f"their tilt angles may be refined, not nominal."
                        )
                    acq_order = get_row_value(row, class_dict, ACQUISITION_ORDER)
                    index = get_row_value(row, class_dict, INDEX)
                    ts_angles.append(
                        ScipionTiltAngle(
                            angle=get_row_value(row, class_dict, TILT_ANGLE),
                            acq_order=int(acq_order) if acq_order is not None else None,
                            index=int(index) if index is not None else None,
                        )
                    )
                angles[ts_id] = ts_angles
        finally:
            conn.close()
        return angles

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
    def _add_ctf_md(
        ti: TiltImage,
        row: sqlite3.Row,
        ts_class_dict: Dict[str, str],
        ctf_md: List[ScipionCtfTomo] | None = None,
    ) -> None:
        """Assigns its CTF to a tilt-image as Scipion does (CTFTomoSeries.getCtfTomoFromTi):
        the one with the same acquisition order or, if the CTFs do not store it, the one
        with the same index. The position in the list is not used: a CTF series may not
        follow the order of the tilt-series or lack some of its tilt-images."""
        ti.ctf_metadata = None
        if not ctf_md:
            return
        match = ScipionSetOfTiltSeries._match_by_acq_order_or_index(
            row, ts_class_dict, ctf_md, "CTFs"
        )
        if match is not None:
            ti.ctf_metadata = match.ctf

    @staticmethod
    def _match_by_acq_order_or_index(
        row: sqlite3.Row,
        ts_class_dict: Dict[str, str],
        items: Sequence[_Matchable],
        description: str,
    ) -> Optional[_Matchable]:
        """Item of ``items`` that corresponds to the tilt-image of ``row``, matched as
        Scipion does (CTFTomoSeries.getCtfTomoFromTi): by acquisition order or, if it is
        not stored in both, by index. None if no item matches.

        :param description: what the items are, for the error messages.
        """
        ts_id = get_row_value(row, ts_class_dict, TS_ID)
        acq_order = get_row_value(row, ts_class_dict, ACQUISITION_ORDER)
        index = get_row_value(row, ts_class_dict, INDEX)
        if acq_order is not None and all(i.acq_order is not None for i in items):
            matches = [i for i in items if i.acq_order == int(acq_order)]
        elif index is not None and all(i.index is not None for i in items):
            matches = [i for i in items if i.index == int(index)]
        else:
            raise ValueError(
                f"tsId = {ts_id}. The {description} cannot be matched with the "
                f"tilt-images: neither the acquisition order nor the index is stored in "
                f"both."
            )
        if len(matches) > 1:
            raise ValueError(
                f"tsId = {ts_id}. Several {description} match the tilt-image of index "
                f"{index} (acquisition order {acq_order})."
            )
        return matches[0] if matches else None

    @staticmethod
    def _build_acquisition(
        row: sqlite3.Row, ts_class_dict: Dict[str, str]
    ) -> Tuple[Instrument, AcquisitionSession]:
        """Builds the dataset-level Instrument and AcquisitionSession from a Scipion
        acquisition row.

        ``voltage`` is a physical instrument property; ``spherical_aberration``,
        ``amplitude_contrast`` and ``tilt_axis_angle`` (degrees, relative to the detector
        x-axis, as in Scipion's ``TomoAcquisition``) are kept at the session level.
        ``get_row_value`` returns None
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
            tilt_axis_angle=get_row_value(row, ts_class_dict, TILT_AXIS_ANGLE),
        )
        return instrument, acquisition_session

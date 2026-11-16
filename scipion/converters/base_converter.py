import ast
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

from cets_data_model.models.models import (
    Affine,
    Translation,
    Scale,
    CoordinateSystem,
    Axis,
    AxisType,
    AxisUnit,
)
from cets_data_model.utils.image_utils import get_mrc_info
from scipion.constants import (
    FILE_NAME,
    OBJECTS_TBL,
    ORIGIN_MATRIX,
    SAMPLING_RATE,
    TOMO_ORIGIN_FIELDS,
    TS_ID,
)
from scipion.utils.utils import validate_file
from scipion.utils.utils_sqlite import connect_db, map_classes_table, get_row_value

# Decimals kept in the coordinates referred to the corner of the tomogram: enough for
# any sub-voxel position, while removing the floating point noise of the change of origin.
COORD_DECIMALS = 6


class BaseConverter:
    def __init__(self, sqlite_path: os.PathLike):
        self.db_path = validate_file(sqlite_path, ".sqlite")
        self.scipion_prj_path = self._get_prj_path()

    def scipion_to_cets(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _get_prj_path(self) -> Path:
        orig_dir = os.getcwd()
        # Move to the project directory and get the full path
        os.chdir(
            self.db_path.parent / ".." / ".."
        )  # PathToScipionUserData/projects/ProjectName/Runs/ProtocolDir/extra/sqlite
        prj_path = os.getcwd()
        # Move to the original working dir
        os.chdir(orig_dir)
        return Path(prj_path)

    def _get_tomo_corner_offset(
        self, tomo_db_path: os.PathLike, tomo_id: str
    ) -> List[float]:
        """Scipion stores the coordinates referred to the origin of their tomogram (by
        default, its center), in voxels. This method returns the vector (x, y, z), in
        voxels, that must be added to them to refer them to the corner of the tomogram,
        that is, to its array coordinate system. It is the same operation Scipion does
        (``Coordinate3D.getPosition(BOTTOM_LEFT_CORNER)``): minus the shifts of the
        origin of the tomogram (Å) divided by its sampling rate. If the tomogram has no
        origin stored, Scipion's default one (the center of the tomogram) is used.

        :param tomo_db_path: sqlite file of the Scipion SetOfTomograms containing the
        tomogram the coordinates are referred to.
        :param tomo_id: Scipion tomogram identifier (tsId).
        """
        tomo_db = validate_file(tomo_db_path, ".sqlite")
        conn = connect_db(tomo_db)
        if conn is None:
            raise Exception(f"Unable to read the tomograms from {tomo_db}")
        try:
            class_dict = map_classes_table(conn)
            sql_fields = self._get_sql_fields(class_dict, TOMO_ORIGIN_FIELDS)
            cursor = conn.cursor()
            cursor.execute(
                f'SELECT {sql_fields} FROM "{OBJECTS_TBL}" WHERE "{class_dict[TS_ID]}"=?',
                (tomo_id,),
            )
            row = cursor.fetchone()
        finally:
            conn.close()
        if row is None:
            raise ValueError(f"The tomogram [{tomo_id}] was not found in {tomo_db}")
        origin_matrix = get_row_value(row, class_dict, ORIGIN_MATRIX)
        sampling_rate = get_row_value(row, class_dict, SAMPLING_RATE)
        if origin_matrix and origin_matrix != "None" and sampling_rate:
            matrix = ast.literal_eval(origin_matrix)
            return [-float(matrix[i][3]) / float(sampling_rate) for i in range(3)]
        # PathToScipionUserData/projects/ProjectName/Runs/ProtocolDir/sqlite
        tomo_prj_path = tomo_db.parents[2]
        return self._get_tomo_center(
            tomo_prj_path / get_row_value(row, class_dict, FILE_NAME)
        )

    @staticmethod
    def _get_tomo_center(tomo_file: os.PathLike) -> List[float]:
        """Center of a tomogram (x, y, z), in voxels: Scipion's default origin."""
        img_info = get_mrc_info(tomo_file)
        return [img_info.size_x / 2, img_info.size_y / 2, img_info.size_z / 2]

    @staticmethod
    def _to_tomo_corner(coords: List[Any], corner_offset: List[float]) -> List[float]:
        """Refers a Scipion coordinate (x, y, z) to the corner of its tomogram."""
        return [
            round(float(coord) + offset, COORD_DECIMALS)
            for coord, offset in zip(coords, corner_offset)
        ]

    @staticmethod
    def _get_sql_fields(mapped_class_dict: Dict[str, str], fields: List[str]) -> str:
        """Gets the mapped fields that will be used in a query formatted as a string.
        :param mapped_class_dict: dict mapping the desired table names with the labelled names,
        e.g. _tsId : c03.
        :param fields: list of the desired labelled names.
        """
        present_fields = [
            f'"{mapped_class_dict[field]}"'
            for field in fields
            if mapped_class_dict.get(field, None)
        ]
        return ", ".join(present_fields)

    @staticmethod
    def _gen_coordinate_systems(
        name: str, ndim: int = 2
    ) -> Tuple[CoordinateSystem, CoordinateSystem]:
        """Builds the (array, physical) coordinate-system pair for an ``ndim`` image/frame.

        Names follow the proposal convention ``{name}_array`` / ``{name}_physical``. The array
        system is pixel/array coords (unit ``pixel``); the physical system is in Å. ``ndim`` is 2
        for 2-D images (x, y) and 3 for volumes (x, y, z).
        """
        axes = ("x", "y", "z")[:ndim]
        array_cs = CoordinateSystem(
            name=f"{name}_array",
            axes=[
                Axis(name=a, axis_type=AxisType.array, axis_unit=AxisUnit.pixel)
                for a in axes
            ],
        )
        physical_cs = CoordinateSystem(
            name=f"{name}_physical",
            axes=[
                Axis(name=a, axis_type=AxisType.space, axis_unit=AxisUnit.angstrom)
                for a in axes
            ],
        )
        return array_cs, physical_cs

    @staticmethod
    def _gen_array_to_physical(
        pixel_size: float,
        array_cs_name: str,
        physical_cs_name: str,
        ndim: int = 2,
    ) -> Scale:
        """The single canonical ``array_to_physical`` transformation for an image: a Scale
        mapping pixel/array coordinates to physical (Å) coordinates by the (isotropic)
        pixel/voxel size. The spec requires exactly one such transformation per image."""
        return Scale(
            scale=[pixel_size] * ndim,
            name="array_to_physical",
            input=array_cs_name,
            output=physical_cs_name,
        )

    @staticmethod
    def _gen_subvolume_transforms(
        euler_matrix: List[List[float]],
        is_coordinate: bool = True,
        pixel_size: float = 1.0,
    ) -> Tuple[Translation, Affine]:
        if is_coordinate:
            name = "Coordinate 3D"
            translation_vector = [0.0, 0.0, 0.0]
        else:
            name = "Subtomogram"
            # Shifts converted from pixels to Å (physical frame).
            translation_vector = [
                euler_matrix[0][-1] * pixel_size,
                euler_matrix[1][-1] * pixel_size,
                euler_matrix[2][-1] * pixel_size,
            ]
        angular_matrix = [
            sublist[:3] for sublist in euler_matrix[:3]
        ]  # Take only the angular 3x3 sub-matrix
        return (
            Translation(
                name="Particle Translation, in angstroms",
                translation=translation_vector,
            ),
            Affine(name=f"{name} orientation", affine=angular_matrix),
        )

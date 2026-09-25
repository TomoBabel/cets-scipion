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
)
from scipion.utils.utils import validate_file


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
        system is pixel/array coords (unitless); the physical system is in Å. ``ndim`` is 2 for
        2-D images (x, y) and 3 for volumes (x, y, z).
        """
        axes = ("x", "y", "z")[:ndim]
        array_cs = CoordinateSystem(
            name=f"{name}_array",
            axes=[Axis(name=a, axis_type=AxisType.array, axis_unit=None) for a in axes],
        )
        physical_cs = CoordinateSystem(
            name=f"{name}_physical",
            axes=[
                Axis(name=a, axis_type=AxisType.space, axis_unit="angstrom")
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

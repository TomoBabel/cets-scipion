import os
from pathlib import Path
from typing import Dict, List, Any

from cets_data_model.models.models import Affine
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
    def _gen_subvolume_transform(
        euler_matrix: List[List[float]], is_coordinate: bool = True
    ) -> Affine:
        angular_matrix = [
            sublist[:3] for sublist in euler_matrix[:3]
        ]  # Take only the angular 3x3 sub-matrix
        name = "Coordinate 3D" if is_coordinate else "Subtomogram"
        return Affine(name=f"{name} orientation", affine=angular_matrix)

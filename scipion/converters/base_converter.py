import os
from pathlib import Path
from typing import Dict, List

from scipion.utils.utils import validate_file


class BaseConverter:
    def __init__(self, sqlite_path: os.PathLike):
        self.db_path = validate_file(sqlite_path, ".sqlite")
        self.scipion_prj_path = self._get_prj_path()

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

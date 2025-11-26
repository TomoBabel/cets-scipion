import os
import traceback
from os import PathLike
from pathlib import Path
from typing import List

import yaml

from cets_data_model.models.models import TiltSeries, Tomogram, Particle3DSet


def validate_file(filename: PathLike, expected_ext: str) -> Path:
    if filename is None:
        raise ValueError("The introduced file cannot be None")
    p = Path(filename).expanduser()
    try:
        p = p.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} does not exist: {p}") from None

    if not p.is_file():
        raise IsADirectoryError(f"{filename} must be a file, not a directory: {p}")

    if not os.access(p, os.R_OK):
        raise PermissionError(f"No read permission for {filename}: {p}")

    ext = p.suffix
    if ext != expected_ext:
        raise ValueError(f"Invalid file extension '{ext}'. Expected: {expected_ext}")
    return p


def validate_new_file(in_file: Path | str) -> Path:
    in_file = Path(in_file).expanduser()
    if in_file.exists():
        in_file.unlink()  # Remove the file
    return in_file


def write_ts_set_yaml(ts_list: List[TiltSeries], output_directory: Path) -> None:
    for ts in ts_list:
        output_file = output_directory / f"tilt_series_{ts.ts_id}_scipion_to_cets.yaml"
        write_obj_yaml(ts, output_file)


def write_tomo_set_yaml(tomo_list: List[Tomogram], output_directory: Path) -> None:
    for tomo in tomo_list:
        output_file = output_directory / f"tomogram_{tomo.tomo_id}_scipion_to_cets.yaml"
        write_obj_yaml(tomo, output_file)


def write_coords_set_yaml(
    coordinates: Particle3DSet, tomo_id: str, output_directory: Path
) -> None:
    output_file = output_directory / f"coordinates_{tomo_id}_scipion_to_cets.yaml"
    write_obj_yaml(coordinates, output_file)


def write_obj_yaml(
    cets_ts_md: TiltSeries | Tomogram | Particle3DSet, yaml_file: Path | str | None
) -> None:
    if yaml_file is None:
        print("write_yaml -> yaml_file is None. Skipping...")
        return
    try:
        yaml_file = validate_new_file(yaml_file)
        metadata_dict = cets_ts_md.model_dump(mode="json")
        with open(yaml_file, "a") as f:
            yaml.dump(metadata_dict, f, sort_keys=False, explicit_start=True)
        print(f"yaml file successfully written! -> {yaml_file}")
    except Exception as e:
        print(
            f"Unable to write the output yaml file {yaml_file} with "
            f"the exception -> {e}"
        )
        print(traceback.format_exc())

"""Command line of the Scipion -> CETS converters (``cets-scipion``).

Scipion projects are only read, so every data type has just a ``to-cets`` sub-command.

Examples::

    cets-scipion tilt-series to-cets \\
        --ts-db Runs/000128_ProtImodImportTransformationMatrix/tiltseries.sqlite \\
        --nominal-ts-db Runs/000002_ProtImportTs/TiltSeries.sqlite \\
        --ctf-db Runs/000084_ProtImportTsCTF/ctftomoseries.sqlite -o ts_cets.yaml
    cets-scipion ctf to-cets --ctf-db ctftomoseries.sqlite --ts-id TS_03 -o ctf.yaml
    cets-scipion tomogram to-cets --tomo-db tomograms.sqlite -o tomos.yaml
    cets-scipion coordinates to-cets --coords-db coordinates3d.sqlite \\
        --tomo-db tomograms.sqlite --pixel-size 10 -o coords.yaml
    cets-scipion subtomograms to-cets --subtomo-db subtomograms.sqlite \\
        --tomo-db tomograms.sqlite --coords-pixel-size 10 -o subtomos.yaml

``to-cets`` writes a CETS yaml file with one entry per kind of object (``tilt_series``,
``alignment``, ``instrument``, ``acquisition_session``, ``ctf``, ``tomogram``,
``coordinates``, ``average``). As a Scipion set holds several tilt-series / tomograms, the
entries are lists.
"""

import argparse
from pathlib import Path
from typing import List, Optional, TypeAlias

from cets_data_model.models.models import Average, PointMatrixSet3D, PointSet3D
from cets_data_model.utils.cli import TO_CETS, add_data_type, add_output_yaml, run_cli
from cets_data_model.utils.yaml_io import write_cets_yaml
from scipion.constants import OBJECTS_TBL, SUBTOMO_ID, TOMO_ID
from scipion.converters.coodinates3d import ScipionSetOfCoordinates3D
from scipion.converters.ctf_set import ScipionSetOfCtf
from scipion.converters.subtomograms import ScipionSetOfSubtomogras
from scipion.converters.tilt_series_set import ScipionSetOfTiltSeries
from scipion.converters.tomograms_set import ScipionSetOfTomograms
from scipion.utils.utils_sqlite import connect_db, map_classes_table

TILT_SERIES_KEY = "tilt_series"
ALIGNMENT_KEY = "alignment"
INSTRUMENT_KEY = "instrument"
ACQUISITION_SESSION_KEY = "acquisition_session"
CTF_KEY = "ctf"
TOMOGRAM_KEY = "tomogram"
COORDINATES_KEY = "coordinates"
AVERAGE_KEY = "average"

DataTypes: TypeAlias = "argparse._SubParsersAction[argparse.ArgumentParser]"


def _add_to_cets(
    data_types: DataTypes, name: str, help_text: str, to_cets_help: str
) -> argparse.ArgumentParser:
    """Adds the sub-command of a data type with its only direction, ``to-cets``."""
    directions = add_data_type(data_types, name, help_text)
    directions.metavar = f"{{{TO_CETS}}}"
    return directions.add_parser(TO_CETS, help=to_cets_help, description=to_cets_help)


def _set_tomo_ids(db_path: Path, tomo_id_field: str) -> List[str]:
    """Identifiers of the tomograms the items of a Scipion set refer to, in order of
    appearance."""
    conn = connect_db(db_path)
    if conn is None:
        raise RuntimeError(f"Unable to read {db_path}")
    try:
        class_dict = map_classes_table(conn)
        if tomo_id_field not in class_dict:
            raise ValueError(f"{db_path} does not store the field {tomo_id_field}.")
        rows = conn.execute(
            f'SELECT "{class_dict[tomo_id_field]}" FROM "{OBJECTS_TBL}"'
        ).fetchall()
    finally:
        conn.close()
    return list(dict.fromkeys(str(row[0]) for row in rows if row[0] is not None))


def _check_tomo_ids(requested: Optional[List[str]], available: List[str]) -> List[str]:
    if not available:
        raise ValueError("The set is empty.")
    if not requested:
        return available
    missing = [tomo_id for tomo_id in requested if tomo_id not in available]
    if missing:
        raise ValueError(
            f"Tomogram id(s) not in the set: {', '.join(missing)}. Available: "
            f"{', '.join(available)}."
        )
    return list(dict.fromkeys(requested))


# ------------------------------------------------------------------------- tilt-series
def tilt_series_to_cets(args: argparse.Namespace) -> None:
    ctf_md = None
    if args.ctf_db is not None:
        ctf_md = ScipionSetOfCtf(args.ctf_db).scipion_to_cets()
        if ctf_md is None:
            raise RuntimeError(f"Unable to read the CTFs from {args.ctf_db}")
    result = ScipionSetOfTiltSeries(args.ts_db).scipion_to_cets(
        ctf_md=ctf_md, nominal_ts_db_path=args.nominal_ts_db
    )
    if result is None:
        raise RuntimeError(f"Unable to read the tilt-series from {args.ts_db}")
    ts_list, alignments, instruments, sessions = result
    out = write_cets_yaml(
        args.output,
        {
            TILT_SERIES_KEY: ts_list,
            ALIGNMENT_KEY: alignments,
            INSTRUMENT_KEY: instruments,
            ACQUISITION_SESSION_KEY: sessions,
        },
    )
    print(f"CETS tilt-series written -> {out}")


def _add_tilt_series(data_types: DataTypes) -> None:
    to_cets = _add_to_cets(
        data_types,
        "tilt-series",
        "Tilt-series: stacks, tilt angles, alignment and acquisition.",
        "Scipion SetOfTiltSeries -> CETS yaml (tilt-series, alignment, instrument, "
        "acquisition session).",
    )
    to_cets.add_argument(
        "--ts-db",
        required=True,
        type=Path,
        help="sqlite file of the SetOfTiltSeries (aligned or not).",
    )
    to_cets.add_argument(
        "--nominal-ts-db",
        type=Path,
        help="sqlite file of the SetOfTiltSeries before the alignment (e.g. the imported "
        "one), to read the nominal tilt angles from. Without it, the nominal tilt angles "
        "of an aligned set are left empty (Scipion only keeps the refined ones).",
    )
    to_cets.add_argument(
        "--ctf-db",
        type=Path,
        help="sqlite file of the SetOfCTFTomoSeries, to add its CTF to each tilt-image "
        "(matched by acquisition order or index).",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=tilt_series_to_cets)


# ------------------------------------------------------------------------- ctf
def ctf_to_cets(args: argparse.Namespace) -> None:
    ctf_md = ScipionSetOfCtf(args.ctf_db).scipion_to_cets()
    if ctf_md is None:
        raise RuntimeError(f"Unable to read the CTFs from {args.ctf_db}")
    if not ctf_md:
        raise ValueError(f"{args.ctf_db} has no CTF series.")
    ts_id = args.ts_id
    if ts_id is None:
        if len(ctf_md) > 1:
            raise ValueError(
                f"{args.ctf_db} has the CTFs of several tilt-series: choose one with "
                f"--ts-id ({', '.join(ctf_md)})."
            )
        ts_id = next(iter(ctf_md))
    if ts_id not in ctf_md:
        raise ValueError(
            f"Tilt-series {ts_id} not in {args.ctf_db}. Available: {', '.join(ctf_md)}."
        )
    out = write_cets_yaml(args.output, {CTF_KEY: [c.ctf for c in ctf_md[ts_id]]})
    print(f"CETS CTF written -> {out}")


def _add_ctf(data_types: DataTypes) -> None:
    to_cets = _add_to_cets(
        data_types,
        "ctf",
        "CTF of the tilt-images of a tilt-series.",
        "Scipion SetOfCTFTomoSeries -> CETS yaml (the CTFs of one tilt-series, in the "
        "order stored in Scipion).",
    )
    to_cets.add_argument(
        "--ctf-db",
        required=True,
        type=Path,
        help="sqlite file of the SetOfCTFTomoSeries.",
    )
    to_cets.add_argument(
        "--ts-id",
        help="Tilt-series to convert (Scipion tsId). Required if the set has several.",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=ctf_to_cets)


# ------------------------------------------------------------------------- tomogram
def tomogram_to_cets(args: argparse.Namespace) -> None:
    tomos = ScipionSetOfTomograms(args.tomo_db).scipion_to_cets()
    if tomos is None:
        raise RuntimeError(f"Unable to read the tomograms from {args.tomo_db}")
    if not tomos:
        raise ValueError(f"{args.tomo_db} has no tomograms.")
    out = write_cets_yaml(args.output, {TOMOGRAM_KEY: tomos})
    print(f"CETS tomograms written -> {out}")


def _add_tomogram(data_types: DataTypes) -> None:
    to_cets = _add_to_cets(
        data_types,
        "tomogram",
        "Tomograms.",
        "Scipion SetOfTomograms -> CETS yaml.",
    )
    to_cets.add_argument(
        "--tomo-db",
        required=True,
        type=Path,
        help="sqlite file of the SetOfTomograms.",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=tomogram_to_cets)


# ------------------------------------------------------------------------- coordinates
def coordinates_to_cets(args: argparse.Namespace) -> None:
    converter = ScipionSetOfCoordinates3D(args.coords_db)
    tomo_ids = _check_tomo_ids(args.tomo_id, _set_tomo_ids(converter.db_path, TOMO_ID))
    point_sets: List[PointSet3D | PointMatrixSet3D] = []
    for tomo_id in tomo_ids:
        point_set = converter.scipion_to_cets(
            tomo_id, args.tomo_db, pixel_size=args.pixel_size
        )
        if point_set is None:
            raise RuntimeError(
                f"Unable to read the coordinates of {tomo_id} from {args.coords_db}"
            )
        point_sets.append(point_set)
    out = write_cets_yaml(args.output, {COORDINATES_KEY: point_sets})
    print(f"CETS coordinates written -> {out}")


def _add_tomo_id(parser: argparse.ArgumentParser, items: str) -> None:
    parser.add_argument(
        "--tomo-id",
        action="append",
        help=f"Tomogram whose {items} are converted (Scipion tsId); repeat it for "
        f"several. Default: all.",
    )


def _add_coordinates(data_types: DataTypes) -> None:
    to_cets = _add_to_cets(
        data_types,
        "coordinates",
        "Picked 3D coordinates.",
        "Scipion SetOfCoordinates3D -> CETS yaml (one PointSet3D per tomogram, or a "
        "PointMatrixSet3D if the coordinates are oriented, in the array (voxel) "
        "coordinate system of the tomogram).",
    )
    to_cets.add_argument(
        "--coords-db",
        required=True,
        type=Path,
        help="sqlite file of the SetOfCoordinates3D.",
    )
    to_cets.add_argument(
        "--tomo-db",
        required=True,
        type=Path,
        help="sqlite file of the SetOfTomograms the coordinates were picked in (their "
        "origin and sampling rate refer the coordinates to the corner of the tomogram).",
    )
    _add_tomo_id(to_cets, "coordinates")
    to_cets.add_argument(
        "--pixel-size",
        type=float,
        help="Voxel size (Å) of the tomograms. With it, the coordinates also carry the "
        "physical (Å) coordinate system and the array_to_physical scale.",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=coordinates_to_cets)


# ------------------------------------------------------------------------- subtomograms
def subtomograms_to_cets(args: argparse.Namespace) -> None:
    converter = ScipionSetOfSubtomogras(args.subtomo_db)
    tomo_ids = _check_tomo_ids(
        args.tomo_id, _set_tomo_ids(converter.db_path, SUBTOMO_ID)
    )
    point_sets: List[PointSet3D | PointMatrixSet3D] = []
    averages: List[Average] = []
    for tomo_id in tomo_ids:
        result = converter.scipion_to_cets(
            tomo_id,
            coords_pixel_size=args.coords_pixel_size,
            tomo_db_path=args.tomo_db,
        )
        if result is None:
            raise RuntimeError(
                f"Unable to read the subtomograms of {tomo_id} from {args.subtomo_db}"
            )
        point_sets.append(result[0])
        averages.append(result[1])
    out = write_cets_yaml(
        args.output, {COORDINATES_KEY: point_sets, AVERAGE_KEY: averages}
    )
    print(f"CETS subtomograms written -> {out}")


def _add_subtomograms(data_types: DataTypes) -> None:
    to_cets = _add_to_cets(
        data_types,
        "subtomograms",
        "Subtomograms (particles) and their picked coordinates.",
        "Scipion SetOfSubTomograms -> CETS yaml (per tomogram, a PointSet3D, or a "
        "PointMatrixSet3D if oriented, with the picked coordinates and an Average with "
        "a ParticleMap per subtomogram).",
    )
    to_cets.add_argument(
        "--subtomo-db",
        required=True,
        type=Path,
        help="sqlite file of the SetOfSubTomograms.",
    )
    to_cets.add_argument(
        "--tomo-db",
        type=Path,
        help="sqlite file of the SetOfTomograms the particles were picked in (to refer "
        "the coordinates to the corner of the tomogram). Default: the origin is the "
        "center of the tomogram of each subtomogram, as in Scipion.",
    )
    _add_tomo_id(to_cets, "subtomograms")
    to_cets.add_argument(
        "--coords-pixel-size",
        type=float,
        help="Voxel size (Å) of the tomograms the particles were picked in (not of the "
        "subtomograms). With it, the coordinates also carry the physical (Å) coordinate "
        "system and the array_to_physical scale.",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=subtomograms_to_cets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cets-scipion",
        description="Converts Scipion sets (sqlite) to CETS metadata (yaml).",
    )
    data_types = parser.add_subparsers(
        dest="data_type",
        metavar="{tilt-series,ctf,tomogram,coordinates,subtomograms}",
        required=True,
    )
    _add_tilt_series(data_types)
    _add_ctf(data_types)
    _add_tomogram(data_types)
    _add_coordinates(data_types)
    _add_subtomograms(data_types)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(build_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())

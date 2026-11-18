import ast
import os
from typing import Dict, List, Tuple

from cets_data_model.models.models import (
    PointSet3D,
    PointMatrixSet3D,
    ParticleMap,
    Average,
    AnnotationType,
)
from cets_data_model.utils.image_utils import get_mrc_info
from scipion.constants import (
    OBJECTS_TBL,
    SUBTOMO_FIELDS,
    SUBTOMO_ID,
    FILE_NAME,
    SUBTOMO_X,
    SUBTOMO_Y,
    SUBTOMO_Z,
    SUBTOMO_TRANSFORM_MATRIX,
    SUBTOMO_COORD_MATRIX,
    SUBTOMO_VOL_NAME,
    SAMPLING_RATE,
)
from scipion.converters.base_converter import BaseConverter
from scipion.utils.utils_sqlite import connect_db, map_classes_table, get_row_value


class ScipionSetOfSubtomogras(BaseConverter):
    def scipion_to_cets(
        self,
        tomo_id: str,
        coords_pixel_size: float | None = None,
        tomo_db_path: os.PathLike | None = None,
        # out_directory: str | None = None
    ) -> Tuple[PointSet3D | PointMatrixSet3D, Average] | None:
        """Converts a set of subtomograms in Scipion sqlite format corresponding to the
        introduced tomogram identifier into CETS metadata.

        In the new data model the old ``Particle3D``/``Particle3DSet`` (which fused a picked
        coordinate and an extracted subvolume into one object) is split into two entities
        (Option A):

        * the picked coordinates -> a ``PointSet3D`` annotation (stored under
          ``Region.annotations``), linked to its tomogram via ``source_tomogram_id``, or a
          ``PointMatrixSet3D`` if they are oriented (see ``_coordinate_rotations``);
        * each extracted subvolume -> a ``ParticleMap`` (stored under ``Average.particle_maps``),
          linked back to a single picked coordinate via ``source_region_id`` +
          ``source_annotation_id`` (the region and the ``PointSet3D`` it lives in) plus
          ``coord_index`` (the 0-based index into ``PointSet3D.origin3D``, so it must stay
          aligned with the order in which the coordinates are appended below).

        This method returns the ``(PointSet3D, Average)`` pair; the caller is responsible for
        placing the ``PointSet3D`` inside the matching ``Region`` and the ``Average`` inside the
        ``Dataset`` (an ``Average`` is not standalone: the referenced region/annotation must be
        resolvable in the same dataset).

        Scipion stores the picked coordinates referred to the origin of their tomogram (by
        default, its center). They are converted here to the array (voxel) coordinate
        system of the tomogram, whose origin is its corner.

        :param tomo_id: Scipion tomogram identifier. It is used to indicate the tomogram from which the
        subtomograms will be converted, as in Scipion the subtomograms from all the tomograms are
        stored together.
        :type tomo_id: str.

        :param coords_pixel_size: voxel size (Å) of the tomogram the particles were picked in.
        The picked coordinates (``origin3D``) are positions in that tomogram's array (voxel)
        frame, so when this is provided the returned ``PointSet3D`` also carries the physical
        (Å) coordinate system and the canonical ``array_to_physical`` scale (making the
        coordinates self-describing). Pass the tomogram's own voxel size so both share the exact
        same physical scale. Note this is the tomogram voxel size, not the extracted-particle
        box voxel size (which is read per particle from its .mrc header for the ParticleMaps).
        When omitted, only the array coordinate system is attached and
        ``coordinate_transformations`` is left empty.
        :type coords_pixel_size: float, optional. Defaults to None.

        :param tomo_db_path: sqlite file of the Scipion SetOfTomograms the particles were
        picked in. The origin and the sampling rate of the tomogram are read from it to
        refer the coordinates to the corner of the tomogram. When omitted, the origin is
        assumed to be the center of the tomogram each subtomogram points to (``_volName``),
        which is Scipion's default.
        :type tomo_db_path: os.PathLike, optional. Defaults to None.
        """
        db_connection = connect_db(self.db_path)
        if db_connection is not None:
            with db_connection as conn:
                # Map the table Classes and get some values from the table Objects
                coord_set_class_dict = map_classes_table(conn)

                # Sqlite fields of the data to be read from each tomogram
                coord_sql_fields = self._get_sql_fields(
                    coord_set_class_dict, SUBTOMO_FIELDS
                )

                cursor = conn.cursor()
                tomo_id_col_name = coord_set_class_dict[SUBTOMO_ID]
                query = f'SELECT {coord_sql_fields} FROM "{OBJECTS_TBL}" WHERE {tomo_id_col_name}="{tomo_id}"'
                cursor.execute(query)  # execute the query

                # TODO (open question #3): id-generation policy. tomo_id is reused as the
                # Region id and as the seed of the annotation id. These must be unique within
                # their respective scopes (Region within Dataset, Annotation within
                # Region.annotations).
                region_id = tomo_id
                annotation_id = f"scipion_coords_{tomo_id}"

                origin_3d = []
                coord_matrices = []
                particle_maps = []
                rows = cursor.fetchall()
                # Vector to refer the coordinates to the corner of the tomogram: read
                # from the tomograms sqlite or, by default, the center of each
                # subtomogram's source tomogram (cached per tomogram file).
                db_corner_offset = (
                    self._get_tomo_corner_offset(tomo_db_path, tomo_id)
                    if tomo_db_path and rows
                    else None
                )
                tomo_centers: Dict[str, List[float]] = {}
                for coord_index, row in enumerate(rows):
                    subtomo_fn = get_row_value(row, coord_set_class_dict, FILE_NAME)
                    subtomo_fn = (
                        self.scipion_prj_path / subtomo_fn
                        if subtomo_fn
                        else self.scipion_prj_path
                    )
                    img_info = get_mrc_info(subtomo_fn)

                    # Every image (particle box) gets an array (voxel, unitless) and a physical
                    # (Å) coordinate system plus exactly one canonical array_to_physical scale
                    # (the box voxel size), per the spec.
                    particle_name = f"particle_{coord_index:03d}"
                    voxel_size = self.resolve_pixel_size(
                        get_row_value(row, coord_set_class_dict, SAMPLING_RATE),
                        img_info.apix_x,
                        f"subtomogram {subtomo_fn}",
                    )
                    array_cs, physical_cs = self._gen_coordinate_systems(
                        particle_name, ndim=3
                    )
                    array_to_physical = self._gen_array_to_physical(
                        voxel_size, array_cs.name, physical_cs.name, ndim=3
                    )

                    # The subtomogram alignment (rotation + shift of the extracted subvolume)
                    # also lives on the ParticleMap, expressed in the physical (Å) frame.
                    subtomo_euler_matrix = ast.literal_eval(
                        get_row_value(
                            row, coord_set_class_dict, SUBTOMO_TRANSFORM_MATRIX
                        )
                    )
                    # Pose reference -> particle: rotation first, then translation
                    # (Scipion stores the inverse; see _gen_subvolume_transforms).
                    subtomo_rot, subtomo_tr = self._gen_subvolume_transforms(
                        subtomo_euler_matrix,
                        is_coordinate=False,
                        pixel_size=voxel_size,
                    )
                    # Anchor the pose (an endomorphism) to the particle physical frame.
                    subtomo_tr.input = physical_cs.name
                    subtomo_tr.output = physical_cs.name
                    subtomo_rot.input = physical_cs.name
                    subtomo_rot.output = physical_cs.name
                    # Orientation of the picked coordinate (see _coordinate_rotations)
                    coord_matrices.append(
                        get_row_value(row, coord_set_class_dict, SUBTOMO_COORD_MATRIX)
                    )
                    if db_corner_offset is not None:
                        corner_offset = db_corner_offset
                    else:
                        vol_name = get_row_value(
                            row, coord_set_class_dict, SUBTOMO_VOL_NAME
                        )
                        if not vol_name:
                            raise ValueError(
                                f"The subtomograms of [{tomo_id}] do not store their "
                                f"source tomogram. Please provide tomo_db_path."
                            )
                        if vol_name not in tomo_centers:
                            tomo_centers[vol_name] = self._get_tomo_center(
                                self.scipion_prj_path / vol_name
                            )
                        corner_offset = tomo_centers[vol_name]
                    origin_3d.append(
                        self._to_tomo_corner(
                            [
                                get_row_value(row, coord_set_class_dict, SUBTOMO_X),
                                get_row_value(row, coord_set_class_dict, SUBTOMO_Y),
                                get_row_value(row, coord_set_class_dict, SUBTOMO_Z),
                            ],
                            corner_offset,
                        )
                    )
                    particle_maps.append(
                        ParticleMap(
                            path=str(subtomo_fn),
                            width=img_info.size_x,
                            height=img_info.size_y,
                            depth=img_info.size_z,
                            # Link this subvolume back to the picked coordinate it was
                            # extracted from: region + PointSet3D annotation + point index.
                            source_region_id=region_id,
                            source_annotation_id=annotation_id,
                            coord_index=coord_index,
                            coordinate_systems=[array_cs, physical_cs],
                            coordinate_transformations=[
                                array_to_physical,
                                subtomo_rot,
                                subtomo_tr,
                            ],
                        )
                    )
                if not particle_maps:
                    raise Exception(
                        f"No particle files were found matching the introduced Scipion's "
                        f"tomogram identifier [{tomo_id}]."
                    )

                # Picked coordinates are positions in the tomogram's array (voxel) frame. When
                # the tomogram voxel size is known we also attach the physical (Å) system and
                # the canonical array_to_physical scale, making the coordinates self-describing.
                # The coordinate-system names match the tomogram's ({tomo_id}_array /
                # {tomo_id}_physical), so both refer to the same frame.
                tomo_array_cs, tomo_physical_cs = self._gen_coordinate_systems(
                    tomo_id, ndim=3
                )
                if coords_pixel_size is not None:
                    coordinate_systems = [tomo_array_cs, tomo_physical_cs]
                    coordinate_transformations = [
                        self._gen_array_to_physical(
                            coords_pixel_size,
                            tomo_array_cs.name,
                            tomo_physical_cs.name,
                            ndim=3,
                        )
                    ]
                else:
                    coordinate_systems = [tomo_array_cs]
                    coordinate_transformations = []
                common_fields = dict(
                    id=annotation_id,
                    name=f"Scipion coordinates for {tomo_id}",
                    source_tomogram_id=tomo_id,
                    origin3D=origin_3d,
                    coordinate_systems=coordinate_systems,
                    coordinate_transformations=coordinate_transformations,
                )
                # Oriented picked coordinates keep their orientations (PointMatrixSet3D)
                rotations = self._coordinate_rotations(coord_matrices)
                point_set: PointSet3D | PointMatrixSet3D
                if rotations is not None:
                    point_set = PointMatrixSet3D(
                        annotation_type=AnnotationType.point_matrix_set_3D,
                        matrix3D=rotations,
                        **common_fields,
                    )
                else:
                    point_set = PointSet3D(
                        annotation_type=AnnotationType.point_set_3D, **common_fields
                    )
                average = Average(
                    name=f"Scipion subtomograms for {tomo_id}",
                    particle_maps=particle_maps,
                )
                # if out_directory:
                #     write_subtomograms_yaml(point_set, average, tomo_id, Path(out_directory))
                return point_set, average
        return None

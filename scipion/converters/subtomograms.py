import ast
from typing import Tuple

from cets_data_model.models.models import (
    PointSet3D,
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
)
from scipion.converters.base_converter import BaseConverter
from scipion.utils.utils_sqlite import connect_db, map_classes_table, get_row_value


class ScipionSetOfSubtomogras(BaseConverter):
    def scipion_to_cets(
        self,
        tomo_id: str,
        # out_directory: str | None = None
    ) -> Tuple[PointSet3D, Average] | None:
        """Converts a set of subtomograms in Scipion sqlite format corresponding to the
        introduced tomogram identifier into CETS metadata.

        In the new data model the old ``Particle3D``/``Particle3DSet`` (which fused a picked
        coordinate and an extracted subvolume into one object) is split into two entities
        (Option A):

        * the picked coordinates -> a ``PointSet3D`` annotation (stored under
          ``Region.annotations``), linked to its tomogram via ``source_tomogram_id``;
        * each extracted subvolume -> a ``ParticleMap`` (stored under ``Average.particle_maps``),
          linked back to a single picked coordinate via ``source_region_id`` +
          ``source_annotation_id`` (the region and the ``PointSet3D`` it lives in) plus
          ``coord_index`` (the 0-based index into ``PointSet3D.origin3D``, so it must stay
          aligned with the order in which the coordinates are appended below).

        This method returns the ``(PointSet3D, Average)`` pair; the caller is responsible for
        placing the ``PointSet3D`` inside the matching ``Region`` and the ``Average`` inside the
        ``Dataset`` (an ``Average`` is not standalone: the referenced region/annotation must be
        resolvable in the same dataset).

        :param tomo_id: Scipion tomogram identifier. It is used to indicate the tomogram from which the
        subtomograms will be converted, as in Scipion the subtomograms from all the tomograms are
        stored together.
        :type tomo_id: str.
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
                particle_maps = []
                for coord_index, row in enumerate(cursor):
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
                    voxel_size = img_info.apix_x if img_info.apix_x else 1.0
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
                    subtomo_tr, subtomo_rot = self._gen_subvolume_transforms(
                        subtomo_euler_matrix,
                        is_coordinate=False,
                        pixel_size=voxel_size,
                    )
                    # Anchor the pose (an endomorphism) to the particle physical frame.
                    subtomo_tr.input = physical_cs.name
                    subtomo_tr.output = physical_cs.name
                    subtomo_rot.input = physical_cs.name
                    subtomo_rot.output = physical_cs.name
                    # TODO (open question #1): the coordinate Euler orientation
                    # (SUBTOMO_COORD_MATRIX) has no home on PointSet3D. It is dropped here;
                    # revisit if per-point orientation of the picked coordinate must be kept.
                    origin_3d.append(
                        [
                            get_row_value(row, coord_set_class_dict, SUBTOMO_X),
                            get_row_value(row, coord_set_class_dict, SUBTOMO_Y),
                            get_row_value(row, coord_set_class_dict, SUBTOMO_Z),
                        ]
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
                                subtomo_tr,
                                subtomo_rot,
                            ],
                        )
                    )
                if not particle_maps:
                    raise Exception(
                        f"No particle files were found matching the introduced Scipion's "
                        f"tomogram identifier [{tomo_id}]."
                    )

                # Picked coordinates are positions in the tomogram's array (voxel) frame.
                tomo_array_cs, _ = self._gen_coordinate_systems(tomo_id, ndim=3)
                point_set = PointSet3D(
                    id=annotation_id,
                    name=f"Scipion coordinates for {tomo_id}",
                    annotation_type=AnnotationType.point_set_3D,
                    source_tomogram_id=tomo_id,
                    origin3D=origin_3d,
                    coordinate_systems=[tomo_array_cs],
                )
                average = Average(
                    name=f"Scipion subtomograms for {tomo_id}",
                    particle_maps=particle_maps,
                )
                # if out_directory:
                #     write_subtomograms_yaml(point_set, average, tomo_id, Path(out_directory))
                return point_set, average
        return None

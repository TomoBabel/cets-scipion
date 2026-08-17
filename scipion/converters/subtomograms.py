import ast
from typing import Tuple

from cets_data_model.models.models import (
    PointSet3D,
    ParticleMap,
    AnnotationReference,
    Average,
    AnnotationType,
    CoordinateSystem,
    Axis,
    AxisType,
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


coordinates_system = [
    CoordinateSystem(
        name="Scipion",
        axes=[Axis(name="ZYZ", axis_type=AxisType.space, axis_unit="pixel")],
    )
]


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
          linked back to a single coordinate via ``source_annotation_reference_id`` +
          ``coord_index``.

        The bridge between the two is an ``AnnotationReference`` inside ``Average.annotations``
        that points at the ``PointSet3D`` (by region id + annotation id). ``coord_index`` is the
        0-based index into ``PointSet3D.origin3D``, so it must stay aligned with the order in
        which the coordinates are appended below.

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
                # Region id and as the seed of the annotation/reference ids. These must be
                # unique within their respective scopes (Region within Dataset, Annotation
                # within Region.annotations, AnnotationReference within Average.annotations).
                annotation_id = f"scipion_coords_{tomo_id}"
                reference_id = f"scipion_ref_{tomo_id}"

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

                    # The subtomogram alignment transform (translation + rotation of the
                    # extracted subvolume) lives on the ParticleMap.
                    subtomo_euler_matrix = ast.literal_eval(
                        get_row_value(
                            row, coord_set_class_dict, SUBTOMO_TRANSFORM_MATRIX
                        )
                    )
                    subtomo_tr, subtomo_rot = self._gen_subvolume_transforms(
                        subtomo_euler_matrix, is_coordinate=False
                    )
                    # Anchor the pose transforms to the declared coordinate system (so
                    # input/output resolve to a real CoordinateSystem on the ParticleMap
                    # instead of being null). The pose is an endomorphism within that frame.
                    cs_name = coordinates_system[0].name
                    subtomo_tr.input = cs_name
                    subtomo_tr.output = cs_name
                    subtomo_rot.input = cs_name
                    subtomo_rot.output = cs_name
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
                            source_annotation_reference_id=reference_id,
                            coord_index=coord_index,
                            # Declare the frame the pose lives in. The axis name ("ZYZ")
                            # surfaces the Euler convention of the affine/translation stored
                            # in coordinate_transformations (the only convention mechanism the
                            # current schema offers; a dedicated ParticleAlignment type + a
                            # rotation_convention field are schema-level, not converter-level).
                            coordinate_systems=coordinates_system,
                            coordinate_transformations=[subtomo_tr, subtomo_rot],
                        )
                    )
                if not particle_maps:
                    raise Exception(
                        f"No particle files were found matching the introduced Scipion's "
                        f"tomogram identifier [{tomo_id}]."
                    )

                point_set = PointSet3D(
                    id=annotation_id,
                    name=f"Scipion coordinates for {tomo_id}",
                    annotation_type=AnnotationType.point_set_3D,
                    source_tomogram_id=tomo_id,
                    origin3D=origin_3d,
                    coordinate_systems=coordinates_system,
                )
                average = Average(
                    name=f"Scipion subtomograms for {tomo_id}",
                    annotations=[
                        AnnotationReference(
                            id=reference_id,
                            source_region_id=tomo_id,
                            source_annotation_id=annotation_id,
                        )
                    ],
                    particle_maps=particle_maps,
                )
                # if out_directory:
                #     write_subtomograms_yaml(point_set, average, tomo_id, Path(out_directory))
                return point_set, average
        return None

# from pathlib import Path
#
# from scipion.converters.coodinates3d import ScipionSetOfCoordinates3D
# from scipion.converters.ctf_set import ScipionSetOfCtf
# from scipion.converters.subtomograms import ScipionSetOfSubtomogras
# from scipion.converters.tilt_series_set import ScipionSetOfTiltSeries
# from scipion.converters.tomograms_set import ScipionSetOfTomograms
# from scipion.utils.utils import write_coords_set_yaml, write_subtomograms_yaml
#
# ### SCIPION TO CETS #################################################################
# # Files
# scratch_dir = "/home/jjimenez/CZII/cets_scratch_dir"
# f_path = Path("/home/jjimenez/ScipionUserData/projects/czii_re5_extract_subtomos/")
# ctf_db_path = f_path / "Runs/000084_ProtImportTsCTF/ctftomoseries.sqlite"
# ts_db_path = f_path / "Runs/000128_ProtImodImportTransformationMatrix/tiltseries.sqlite"
# tomo_db_path = f_path / "Runs/000821_ProtImportTomograms/tomograms.sqlite"
# coords_db_path = (
#     f_path / "Runs/000872_ProtImportCoordinates3DFromStar/coordinates3d.sqlite"
# )
# subtomo_db_path = f_path / "Runs/001008_DynamoSubTomoMRA/subtomograms.sqlite"
#
# # CTF metadata
# sci_ctf_set = ScipionSetOfCtf(ctf_db_path)
# ctf_md_dict = sci_ctf_set.scipion_to_cets()
#
# # TS Metadata
# sci_ts_set = ScipionSetOfTiltSeries(ts_db_path)
# sci_ts_set.scipion_to_cets(ctf_md=ctf_md_dict, out_directory=scratch_dir)
#
# # Tomogram metadata (tomogram-only; particles are handled separately below)
# sci_tomo_set = ScipionSetOfTomograms(tomo_db_path)
# tomo_md_list = sci_tomo_set.scipion_to_cets(out_directory=scratch_dir)
#
# # Coordinates -> one PointSet3D annotation per tomogram (Option B).
# # In the data model each PointSet3D goes under the matching Region.annotations.
# sci_coords_set = ScipionSetOfCoordinates3D(coords_db_path)
# for tomo in tomo_md_list:
#     tomo_id = tomo.tilt_series_id
#     point_set = sci_coords_set.scipion_to_cets(tomo_id)
#     if point_set is not None:
#         write_coords_set_yaml(point_set, tomo_id, Path(scratch_dir))
#
# # Subtomograms -> (PointSet3D, Average) per tomogram (Option A). The PointSet3D holds the
# # coordinates (Region.annotations) and the Average holds the extracted ParticleMaps
# # (Dataset.averages), linked via AnnotationReference + coord_index.
# sci_subtomo_set = ScipionSetOfSubtomogras(subtomo_db_path)
# for tomo in tomo_md_list:
#     tomo_id = tomo.tilt_series_id
#     result = sci_subtomo_set.scipion_to_cets(tomo_id)
#     if result is not None:
#         subtomo_point_set, average = result
#         write_subtomograms_yaml(subtomo_point_set, average, tomo_id, Path(scratch_dir))

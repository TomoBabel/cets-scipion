from pathlib import Path

from scipion.converters.ctf_set import ScipionSetOfCtf
from scipion.converters.tilt_series_set import ScipionSetOfTiltSeries
from scipion.converters.tomograms_set import ScipionSetOfTomograms

### SCIPION TO CETS #################################################################
# Files
scratch_dir = "/home/jjimenez/CZII/cets_scratch_dir"
f_path = Path("/home/jjimenez/ScipionUserData/projects/czii_re5_extract_subtomos/")
ctf_db_path = f_path / "Runs/000084_ProtImportTsCTF/ctftomoseries.sqlite"
ts_db_path = f_path / "Runs/000128_ProtImodImportTransformationMatrix/tiltseries.sqlite"
tomo_db_path = f_path / "Runs/000821_ProtImportTomograms/tomograms.sqlite"
coords_db_path = (
    f_path / "Runs/000872_ProtImportCoordinates3DFromStar/coordinates3d.sqlite"
)
subtomo_db_paht = f_path / "Runs/001008_DynamoSubTomoMRA/subtomograms.sqlite"

# CTF metadata
sci_ctf_set = ScipionSetOfCtf(ctf_db_path)
ctf_md_dict = sci_ctf_set.scipion_to_cets()

# TS Metadata
sci_ts_set = ScipionSetOfTiltSeries(ts_db_path)
sci_ts_set.scipion_to_cets(ctf_md=ctf_md_dict, out_directory=scratch_dir)

# Tomogram metadata with coordinates
sci_tomo_set = ScipionSetOfTomograms(tomo_db_path)
tomo_md_list = sci_tomo_set.scipion_to_cets(
    particles_db_path=coords_db_path, out_directory=scratch_dir
)

# Tomogram metadata with subtomograms
sci_tomo_set = ScipionSetOfTomograms(tomo_db_path)
tomo_md_list = sci_tomo_set.scipion_to_cets(
    particles_db_path=subtomo_db_paht, out_directory=scratch_dir
)

from pathlib import Path

from scipion.converters.ctf_set import ScipionSetOfCtf
from scipion.converters.tilt_series_set import ScipionSetOfTiltSeries


### SCIPION TO CETS #################################################################
# Files
scratch_dir = "/home/jjimenez/CZII/cets_scratch_dir"
f_path = Path(
    "/home/jjimenez/ScipionUserData/projects/chlamy/Runs/003123_ProtAreTomoAlignRecon/"
)
ctf_db_path = f_path / "ctftomoseries.sqlite"
ts_db_fn = f_path / "tiltseries.sqlite"

# CTF metadata
sci_ctf_set = ScipionSetOfCtf(ctf_db_path)
ctf_md_dict = sci_ctf_set.scipion_to_cets()

# TS Metadata
sci_ts_set = ScipionSetOfTiltSeries(ts_db_fn)
sci_ts_set.scipion_to_cets(ctf_md=ctf_md_dict, out_directory=scratch_dir)

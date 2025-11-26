# Master table names
CLASSES_TBL = "Classes"
OBJECTS_TBL = "Objects"
PROPERTIES_TBL = "Properties"

# Master table columns
LABEL_PROPERTY = "label_property"
COLUMN_NAME = "column_name"
PROP_KEY = "key"
PROP_VALUE = "value"

# TILT-SERIES ##########################################
# Field names
TS_ID = "_tsId"
FILE_NAME = "_filename"
INDEX = "_index"
ACQUISITION_ORDER = "_acqOrder"
TILT_ANGLE = "_tiltAngle"
ACCUMULATED_DOSE = "_acquisition._accumDose"
TRANSFORMATION_MATRIX = "_transform._matrix"
ODD_EVEN_FN = "_oddEvenFileNames"

TILT_SERIES_FIELDS = [
    TS_ID,
    FILE_NAME,
    INDEX,
    ACQUISITION_ORDER,
    TILT_ANGLE,
    ACCUMULATED_DOSE,
    TRANSFORMATION_MATRIX,
    ODD_EVEN_FN,
]

CTF_CORRECTED = "_ctfCorrected"

# CTF ###############################################
DEFOCUS_U = "_defocusU"
DEFOCUS_V = "_defocusV"
DEFOCUS_ANGLE = "_defocusAngle"
PHASE_SHIFT = "_phaseShift"

CTF_TOMO_SERIES_FIELDS = [
    DEFOCUS_U,
    DEFOCUS_V,
    DEFOCUS_ANGLE,
    PHASE_SHIFT,
    ACQUISITION_ORDER,
]

# TOMOGRAM ###########################################
ODD_EVEN_TOMOS_FN = "_halfMapFilenames"

TOMOGRAM_FIELDS = [TS_ID, FILE_NAME, CTF_CORRECTED, ODD_EVEN_TOMOS_FN]

# COORDINATES 3D #####################################
COORD_X = "_x"
COORD_Y = "_y"
COORD_Z = "_z"
TOMO_ID = "_tomoId"
EULER_MATRIX = "_eulerMatrix._matrix"

COORD_3D_FIELDS = [COORD_X, COORD_Y, COORD_Z, TOMO_ID, EULER_MATRIX]

# SUBTOMOGRAMS ########################################
SUBTOMO_X = "_coordinate._x"
SUBTOMO_Y = "_coordinate._y"
SUBTOMO_Z = "_coordinate._z"
SUBTOMO_COORD_MATRIX = "_coordinate._eulerMatrix._matrix"
SUBTOMO_TRANSFORM_MATRIX = "_transform._matrix"
SUBTOMO_ID = "_coordinate._tomoId"

SUBTOMO_FIELDS = [
    FILE_NAME,
    SUBTOMO_X,
    SUBTOMO_Y,
    SUBTOMO_Z,
    SUBTOMO_COORD_MATRIX,
    SUBTOMO_TRANSFORM_MATRIX,
    SUBTOMO_ID,
]

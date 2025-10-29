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

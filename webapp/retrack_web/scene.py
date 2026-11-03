"""
The configured scene: the WILDTRACK-style seven-camera rig (SRS 2.4, C-01 to C-05).

Values that describe the model's scene configuration mirror WorldTrack/configs/d_wildtrack.yml and
WorldTrack/datasets/wildtrack_dataset.py, which are what the trained checkpoints were fitted to.
"""

RIG_ID = 'RIG-WILDTRACK-7'
RIG_NAME = 'WILDTRACK seven-camera rig'

# C-01: exactly seven fixed cameras in this order
CAMERA_ORDER = ('CVLab1', 'CVLab2', 'CVLab3', 'CVLab4', 'IDIAP1', 'IDIAP2', 'IDIAP3')
CAMERA_COUNT = len(CAMERA_ORDER)

# C-03: accepted images are 1920 by 1080 pixels
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# C-02 / NFR-15: one shared ground-plane convention, centimetres
COORDINATE_CONVENTION = 'WILDTRACK_GROUND_PLANE'
COORDINATE_CONVENTION_DESCRIPTION = ('WILDTRACK world coordinates: ground plane z = 0, x and y in centimetres, '
                                     'origin and axes as defined by the WILDTRACK calibration')
TRANSLATION_UNIT = 'cm'

# Scene configuration a checkpoint must have been trained with (C-05)
MODEL_NUM_CAMERAS = CAMERA_COUNT
MODEL_RESOLUTION = (120, 4, 360)  # Y, Z, X
MODEL_BOUNDS = (0, 1440, 0, 480, 0, 200)  # xmin, xmax, ymin, ymax, zmin, zmax in 2.5 cm grid cells

# WILDTRACK ground grid (2.5 cm cells) -> world coordinates in cm
WORLDCOORD_FROM_WORLDGRID = ((0.0, 2.5, -300.0),
                             (2.5, 0.0, -900.0),
                             (0.0, 0.0, 1.0))

# Ground-plane area covered by the grid, in world cm (used to draw the bird's eye view)
GROUND_X_RANGE_CM = (-300.0, 900.0)
GROUND_Y_RANGE_CM = (-900.0, 2700.0)


def worldgrid_to_world_cm(grid_x, grid_y):
    m = WORLDCOORD_FROM_WORLDGRID
    return (m[0][0] * grid_x + m[0][1] * grid_y + m[0][2],
            m[1][0] * grid_x + m[1][1] * grid_y + m[1][2])

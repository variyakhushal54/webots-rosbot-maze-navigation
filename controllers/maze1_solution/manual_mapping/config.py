# This file keeps the settings used by mapping, planning, and movement.
# Change values here to tune the robot without editing the main control code.
# Distances are measured in metres unless a comment or variable name says otherwise.

import math

# Turns main visualization on or off.
ENABLE_MAIN_VIZ  = True
# Turns temp visualization on or off.
ENABLE_TEMP_VIZ  = False
# Turns costmap visualization on or off.
ENABLE_COSTMAP_VIZ = False


# Number of cells along each side of the square occupancy map.
MAP_SIZE     = 1000
# Real-world width and height of one map cell, in metres.
MAP_RES      = 0.02
# Largest absolute occupancy score stored in a map cell.
LOG_CLIP     = 10.0
# Occupancy score that makes a confirmed wall permanent.
WALL_LOCK    = 10.0
# Minimum occupancy score required to treat a cell as a wall.
WALL_THRESH  = 3.0
# Maximum occupancy score used to treat a cell as known free space.
FREE_THRESH  = -0.5
# Longest sensor range accepted by the mapping code, in metres.
MAX_RANGE    = 12.0
# Update interval used for floating wall detect interval, in frames.
FLOATING_WALL_DETECT_EVERY  = 6
# Controls lidar sampling step.
LIDAR_SUBSAMPLE             = 1


# Distance used for lidar frontier range, in metres.
LIDAR_FRONTIER_RANGE        = 6.0
# Controls lidar free decrement.
LIDAR_FREE_DECREMENT        = 1.50
# Controls lidar wall increment.
LIDAR_WALL_INCREMENT        = 2.00
# Controls lidar post turn settle.
LIDAR_POST_TURN_SETTLE      = 5


# Turns depth-first search lighthouse exploration on or off.
USE_DFS_LIGHTHOUSE_EXPLORATION = False


# Turns pose graph on or off.
USE_POSE_GRAPH      = True
# Distance used for node spacing, in metres.
NODE_SPACING_M      = 0.20
# Distance used for loop radius, in metres.
LOOP_RADIUS_M       = 0.50
# Controls loop time gap.
LOOP_TIME_GAP       = 30
# Score or fraction used for loop fitness threshold.
LOOP_FITNESS_THRESH = 0.50


# Radius of each drive wheel, in metres.
WHEEL_RADIUS = 0.0425
# Distance between the left and right wheels, in metres.
WHEEL_TREAD  = 0.165

# Half of the robot body length, including its safety buffer, in metres.
ROBOT_HALF_L = 0.120
# Half of the robot body width, including its safety buffer, in metres.
ROBOT_HALF_W = 0.150
# Full robot body length used for collision checks, in metres.
ROBOT_LENGTH = 0.24
# Full robot body width used for collision checks, in metres.
ROBOT_WIDTH  = 0.30
# Robot height used to decide whether it can pass under an obstacle, in metres.
ROBOT_HEIGHT = 0.25

# Legacy pillar radius used by older calculations, in metres.
PILLAR_RADIUS = 0.05

# Expected physical height of a pillar, in metres.
PILLAR_HEIGHT_M = 0.30
# Expected physical radius of a pillar, in metres.
PILLAR_RADIUS_M = 0.10
# Expected physical width of a pillar, in metres.
PILLAR_DIAMETER_M = 2.0 * PILLAR_RADIUS_M

# Distance from a pillar at which the robot counts as having arrived, in metres.
PILLAR_ARRIVE_DIST = ROBOT_HALF_L + PILLAR_RADIUS_M


# Distance used for blue touch sensor distance, in metres.
BLUE_TOUCH_SENSOR_DIST = 0.05
# Time allowed for blue touch, in seconds.
BLUE_TOUCH_TIMEOUT     = 6.0


# Score or fraction used for pillar visibility minimum fraction.
PILLAR_VISIBILITY_MIN_FRAC   = 0.55
# Distance used for pillar occlusion maximum distance, in metres.
PILLAR_OCCLUSION_MAX_DIST    = 3.5
# Distance used for pillar occlusion minimum distance, in metres.
PILLAR_OCCLUSION_MIN_DIST    = 0.35


# Score or fraction used for pillar occlusion stamp fraction.
PILLAR_OCCLUSION_STAMP_FRAC  = 0.35
# Number of update frames used for pillar occlusion minimum.
PILLAR_OCCLUSION_MIN_FRAMES  = 3
# Camera height above the ground, in metres.
CAM_H        = 0.188
# Expected height of the top of a normal wall, in metres.
WALL_TOP_H   = 0.75

# Controls safe front.
SAFE_FRONT   = ROBOT_HALF_L + 0.05
# Controls safe side.
SAFE_SIDE    = ROBOT_HALF_W + 0.05
# Controls safe diagonal.
SAFE_DIAG    = ROBOT_HALF_W + 0.04


# Distance used for forward margin, in metres.
FWD_MARGIN   = 0.08
# Controls hard stop.
HARD_STOP    = 0.03
# Distance used for turn margin, in metres.
TURN_MARGIN  = 0.03


# Controls infrared corner warn limit.
IR_CORNER_WARN_LIMIT = 0.08
# Controls infrared corner hard stop.
IR_CORNER_HARD_STOP  = 0.03


# Controls lidar offset x.
LIDAR_OFFSET_X   =  0.020
# Controls lidar offset y.
LIDAR_OFFSET_Y   =  0.000
# Controls lidar offset height.
LIDAR_OFFSET_Z   =  0.180
# Controls lidar offset yaw.
LIDAR_OFFSET_YAW =  0.000

# Position and viewing angle of each infrared sensor relative to the robot centre.
IR_OFFSETS = {
    'fl': ( 0.105,  0.080, 0.055,  math.pi / 4),
    'fr': ( 0.105, -0.080, 0.055, -math.pi / 4),
    'rl': (-0.105,  0.080, 0.055,  3 * math.pi / 4),
    'rr': (-0.105, -0.080, 0.055, -3 * math.pi / 4),
}

# Controls camera offset x.
CAMERA_OFFSET_X   =  0.000
# Controls camera offset y.
CAMERA_OFFSET_Y   =  0.011
# Controls camera offset height.
CAMERA_OFFSET_Z   =  0.199
# Controls camera offset pitch.
CAMERA_OFFSET_PITCH = 0.0
# Controls camera offset yaw.
CAMERA_OFFSET_YAW   = 0.0
# Controls depth offset x.
DEPTH_OFFSET_X    =  0.000
# Controls depth offset y.
DEPTH_OFFSET_Y    =  0.037
# Controls depth offset height.
DEPTH_OFFSET_Z    =  0.199
# Distance used for maximum ground distance, in metres.
MAX_GROUND_DIST     = 3.0


# Lowest HSV colour values accepted as green floor.
GREEN_HSV_LOW   = (38, 80, 50)
# Highest HSV colour values accepted as green floor.
GREEN_HSV_HIGH  = (90, 255, 255)


# Distance used for green minimum stamp distance, in metres.
GREEN_MIN_STAMP_DIST  = 0.20
# Distance used for green maximum stamp distance, in metres.
GREEN_MAX_STAMP_DIST  = 1.5


# Score or fraction used for green row minimum fraction.
GREEN_ROW_MIN_FRAC   = 0.55
# Score or fraction used for green row maximum fraction.
GREEN_ROW_MAX_FRAC   = 0.92


# Controls green depth bias.
GREEN_DEPTH_BIAS     = 0.0


# Minimum value required for green vote threshold.
GREEN_VOTE_THRESHOLD  = 3
# Update interval used for green vote decay, in frames.
GREEN_VOTE_DECAY      = 60


# Map-cell value used for green stamp radius.
GREEN_STAMP_RADIUS    = 1


# Controls green sampling step stride.
GREEN_SUBSAMPLE_STRIDE = 3


# Distance used for pillar detect range, in metres.
PILLAR_DETECT_RANGE = 4.0
# Distance used for pillar confirm distance, in metres.
PILLAR_CONFIRM_DIST = 1.5
# Controls pillar navigation minimum buffer.
PILLAR_NAV_MIN_BUF  = 5
# Distance used for close approach distance, in metres.
CLOSE_APPROACH_DIST = 0.50
# Distance used for close confirm distance, in metres.
CLOSE_CONFIRM_DIST  = ROBOT_HALF_L + PILLAR_RADIUS_M + 0.01
# Distance used for green obstacle confirm distance, in metres.
GREEN_OBSTACLE_CONFIRM_DIST = 1.2
# Controls green half.
GREEN_HALF = 0.25

# Minimum value required for narrow gap threshold.
NARROW_GAP_THRESHOLD = 0.90
# Turns gap closing on or off.
ENABLE_GAP_CLOSING = False


# Pixel value used for pillar tiny minimum pixels.
PILLAR_TINY_MIN_PX   = 4
# Pixel value used for pillar visual lock maximum pixels.
PILLAR_VLOCK_MAX_PX  = 3000
# Time allowed for visual lock, in seconds.
VISUAL_LOCK_TIMEOUT  = 8.0
# Time allowed for visual lock, in seconds.
VISUAL_LOCK_COOLDOWN = 10.0


# Turns junction scan on or off.
ENABLE_JUNCTION_SCAN = False


# Turns narrow gap walls on or off.
ENABLE_NARROW_GAP_WALLS      = True


# Turns camera debug on or off.
SHOW_CAMERA_DEBUG = False


# Controls frontier center snap.
FRONTIER_CENTER_SNAP = True


# State name used while the controller is starting.
STATE_STARTUP    = 'STARTUP'
# State name used for the first scan of the surroundings.
STATE_PRESCAN    = 'PRESCAN'
# State name used while searching unknown parts of the maze.
STATE_EXPLORE    = 'EXPLORE'
# State name used while travelling to the blue pillar.
STATE_NAV_BLUE   = 'NAV_BLUE'
# State name used while travelling to the yellow pillar.
STATE_NAV_YELLOW = 'NAV_YELLOW'
# State name used while looking for a mission target.
STATE_SEARCH_TARGET = 'SEARCH_TARGET'
# State name used while checking a possible target.
STATE_LOOK_AT_CANDIDATE = 'LOOK_AT_CANDIDATE'
# State name used after the mission has finished.
STATE_DONE       = 'DONE'


# Distance used for path draw width, in metres.
PATH_DRAW_WIDTH = 0.16
# Controls path arrive tolerance.
PATH_ARRIVE_TOL = 0.08
# Controls path skip tolerance.
PATH_SKIP_TOL   = 0.15
# Distance used for path lookahead, in metres.
PATH_LOOKAHEAD  = 0.40

# Clearance levels tried by the planner, from safest to most permissive.
PLANNER_KERNELS = [11, 9, 7, 5, 3, 1, 0]

# Controls how strongly A-star hug affects route or detection scoring.
ASTAR_HUG_WEIGHT       = 45.0
# Controls how strongly A-star narrow affects route or detection scoring.
ASTAR_NARROW_WEIGHT    = 2.0
# Controls how strongly A-star turn affects route or detection scoring.
ASTAR_TURN_WEIGHT      = 2.0
# Controls A-star gaussian spread.
ASTAR_GAUSSIAN_STD     = 0.15
# Controls how strongly A-star centering affects route or detection scoring.
ASTAR_CENTERING_WEIGHT = 150.0
# Controls how strongly A-star visit affects route or detection scoring.
ASTAR_VISIT_WEIGHT     = 0.5
# Controls how strongly A-star trail affects route or detection scoring.
ASTAR_TRAIL_WEIGHT     = 1.0
# Controls how strongly A-star geometry affects route or detection scoring.
ASTAR_GEOM_WEIGHT      = 3.0
# Controls how strongly A-star corner affects route or detection scoring.
ASTAR_CORNER_WEIGHT    = 5.0
# Controls how strongly frontier width affects route or detection scoring.
FRONTIER_WIDTH_WEIGHT  = 1.8
# Controls how strongly frontier novelty affects route or detection scoring.
FRONTIER_NOVELTY_WEIGHT = 20.0
# Controls how strongly frontier distance affects route or detection scoring.
FRONTIER_DISTANCE_WEIGHT = 20.0


# Maximum forward speed allowed by the motion controller, in metres per second.
CTRL_MAX_VX         = 0.60
# Maximum turning speed allowed by the motion controller, in radians per second.
CTRL_MAX_OM         = 4.0
# Maximum wheel rotation speed, in radians per second.
CTRL_MAX_WHEEL_RAD  = 10.0

# Forward speed used for manual driving, in metres per second.
MANUAL_VX_FWD = 0.41
# Reverse speed used for manual driving, in metres per second.
MANUAL_VX_REV = -0.30
# Turning speed used for manual driving, in radians per second.
MANUAL_OM     = 2.5

# Controls approach maximum turning speed.
APPROACH_MAX_OM     = 3.5
# Controls approach tight-space turning speed.
APPROACH_TIGHT_OM   = 1.2
# Controls approach turning speed gain.
APPROACH_OM_GAIN    = 4.0
# Controls approach steer gain.
APPROACH_STEER_GAIN = 2.5

# Controls follow speed normal.
FOLLOW_SPEED_BASE      = 0.40
# Controls follow speed tight-space.
FOLLOW_SPEED_TIGHT     = 0.28
# Distance used for follow lookahead normal, in metres.
FOLLOW_LOOKAHEAD_BASE  = 0.20
# Distance used for follow lookahead tight-space, in metres.
FOLLOW_LOOKAHEAD_TIGHT = 0.15
# Controls follow turning speed gain.
FOLLOW_OM_GAIN         = 2.6

# Number of controller updates used for recovery stuck.
RECOVERY_STUCK_TICKS = 80
# Distance used for recovery stuck distance, in metres.
RECOVERY_STUCK_DIST  = 0.01
# Controls recovery timeout l2.
RECOVERY_TIMEOUT_L2  = 4.0
# Distance used for recovery escape distance, in metres.
RECOVERY_ESCAPE_DIST = 0.30


# Controls navigation loop breaker escalate.
NAV_LOOP_BREAKER_ESCALATE    = 3
# Number of controller updates used for navigation recovery suppress.
NAV_RECOVERY_SUPPRESS_TICKS  = 120


# Distance used for visual range, in metres.
VISUAL_RANGE           = 0.65
# Time allowed for explore hard, in seconds.
EXPLORE_HARD_TIMEOUT   = 180.0
# Controls explore cluster r.
EXPLORE_CLUSTER_R      = 15
# Update interval used for explore replan interval, in frames.
EXPLORE_REPLAN_EVERY   = 100
# Minimum value required for breadth-first search strike threshold.
BFS_STRIKE_THRESHOLD   = 3
# Distance used for junction reach distance, in metres.
JUNCTION_REACH_DIST    = 0.45
# Distance used for junction cluster distance, in metres.
JUNCTION_CLUSTER_DIST  = 1.50


# Distance used for coordinate record distance, in metres.
COORD_RECORD_DIST = 0.20
# Controls coordinate history maximum.
COORD_HIST_MAX    = 500


# Lowest HSV colour values accepted as wood.
WOOD_HSV_LOW               = (8, 60, 70)
# Highest HSV colour values accepted as wood.
WOOD_HSV_HIGH              = (28, 255, 255)
# Controls floating depth bias.
FLOATING_DEPTH_BIAS        = 0.0
# Distance used for height margin below, in metres.
Z_MARGIN_BELOW             = 0.04


# Distance used for height ceiling, in metres.
Z_CEILING                  = 0.66
# Distance used for maximum detect distance, in metres.
MAX_DETECT_DIST            = 3.2
# Update interval used for floating wall detect interval, in frames.
FLOATING_WALL_DETECT_EVERY = 6


# Controls floating wall danger minimum.
FLOATING_WALL_DANGER_MIN   = 0.03
# Controls floating wall danger maximum.
FLOATING_WALL_DANGER_MAX   = 0.40
# Controls floating wall danger count.
FLOATING_WALL_DANGER_COUNT = 35
# Controls floating wall log increase.
FLOATING_WALL_LOG_INCR     = 1.5
# Map-cell value used for floating wall radius cells.
FLOATING_WALL_RADIUS_CELLS = 2

# Controls floating height floor guard.
FLOATING_Z_FLOOR_GUARD     = 0.015


# Distance used for floating height ceiling, in metres.
FLOATING_Z_CEILING         = 0.26
# Distance used for floating height ceiling relaxed, in metres.
FLOATING_Z_CEILING_RELAXED = 0.80
# Minimum value required for floating height blocking threshold.
FLOATING_Z_BLOCKING_THRESHOLD = 0.21
# Distance used for floating maximum distance, in metres.
FLOATING_MAX_DIST          = 2.0
# Pixel value used for floating vertical k pixels.
FLOATING_VERTICAL_K_PIXELS = 6
# Controls floating wall minimum points.
FLOATING_WALL_MIN_POINTS   = 5
# Controls floating wall connect gap.
FLOATING_WALL_CONNECT_GAP  = 5
# Controls floating fov guard.
FLOATING_FOV_GUARD         = 0.7
# Number of update frames used for floating wall decay.
FLOATING_WALL_DECAY_FRAMES = 60


# Minimum value required for floating vote threshold.
FLOATING_VOTE_THRESHOLD    = 4
# Number of update frames used for floating vote decay.
FLOATING_VOTE_DECAY_FRAMES = 50
# Distance used for floating minimum wall width, in metres.
FLOATING_MIN_WALL_WIDTH_M  = 0.15


# Map-cell value used for floating lidar strip margin cells.
FLOATING_LIDAR_STRIP_MARGIN_CELLS = 10
# Distance used for floating wall minimum distance, in metres.
FLOATING_WALL_MIN_DIST_M = 0.10


# Map-cell value used for floating noise minimum area cells.
FLOATING_NOISE_MIN_AREA_CELLS = 18
# Minimum value required for floating noise blob threshold.
FLOATING_NOISE_BLOB_THRESH    = 40
# Controls floating noise minimum aspect.
FLOATING_NOISE_MIN_ASPECT     = 1.8


# Controls floating stuck lidar open.
FLOATING_STUCK_LIDAR_OPEN     = 0.30
# Distance used for inferred floating width, in metres.
INFERRED_FLOATING_WIDTH       = ROBOT_WIDTH + 0.12
# Distance used for inferred floating ahead, in metres.
INFERRED_FLOATING_AHEAD       = 0.14
# Controls inferred floating number of points.
INFERRED_FLOATING_N_POINTS    = 15
# Map-cell value used for inferred floating radius cells.
INFERRED_FLOATING_RADIUS_CELLS = 3
# Distance used for inferred floating forward reach, in metres.
INFERRED_FLOATING_FORWARD_REACH = 0.28
# Controls inferred floating forward steps.
INFERRED_FLOATING_FORWARD_STEPS = 5


# Pixel value used for floating weak wood pixels.
FLOATING_WEAK_WOOD_PIXELS          = 80
# Controls floating depth only minimum points.
FLOATING_DEPTH_ONLY_MIN_POINTS     = 12
# Score or fraction used for floating fusion depth danger fraction.
FLOATING_FUSION_DEPTH_DANGER_FRAC  = 0.25
# Controls how strongly floating fusion depth affects route or detection scoring.
FLOATING_FUSION_DEPTH_WEIGHT       = 1.0
# Controls how strongly floating fusion wood affects route or detection scoring.
FLOATING_FUSION_WOOD_WEIGHT        = 0.9
# Controls how strongly floating fusion stuck affects route or detection scoring.
FLOATING_FUSION_STUCK_WEIGHT       = 1.2
# Controls how strongly floating fusion lidar open affects route or detection scoring.
FLOATING_FUSION_LIDAR_OPEN_WEIGHT  = 0.6
# Controls how strongly floating fusion tactile affects route or detection scoring.
FLOATING_FUSION_TACTILE_WEIGHT     = 1.0
# Score or fraction used for floating fusion camera stamp score.
FLOATING_FUSION_CAMERA_STAMP_SCORE = 1.85
# Score or fraction used for floating fusion infer stamp score.
FLOATING_FUSION_INFER_STAMP_SCORE  = 2.0
# Controls how strongly floating fusion pillar affects route or detection scoring.
FLOATING_FUSION_PILLAR_WEIGHT      = 1.1


# Distance used for floating wall confirm distance, in metres.
FLOATING_WALL_CONFIRM_DIST_M       = 0.55
# Number of update frames used for floating wall pending maximum.
FLOATING_WALL_PENDING_MAX_FRAMES   = 120

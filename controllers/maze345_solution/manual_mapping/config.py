# This file contains the values that control the complete mapping mission.
# Sensor geometry describes where each device sits on the robot. The remaining
# groups tune perception, map updates, path planning, movement, and recovery.
# Changing a value here changes behavior without changing the related algorithm.

import math

# Enables main visualization when set to true.
ENABLE_MAIN_VIZ  = True
# Enables temp visualization when set to true.
ENABLE_TEMP_VIZ  = False
# Enables costmap visualization when set to true.
ENABLE_COSTMAP_VIZ = False


# Sets the width and height of the square occupancy grid in cells.
MAP_SIZE     = 1000
# Sets the real-world size of one map cell in metres.
MAP_RES      = 0.02
# Limits occupancy log-odds so repeated readings cannot grow without bound.
LOG_CLIP     = 10.0
# Marks cells at this log-odds value as permanent walls that free-space rays cannot erase.
WALL_LOCK    = 10.0
# Treats cells at or above this log-odds value as occupied.
WALL_THRESH  = 3.0
# Treats cells at or below this log-odds value as confirmed free space.
FREE_THRESH  = -0.5
# Sets the longest lidar distance, in metres, accepted for mapping.
MAX_RANGE    = 12.0
# Initial floating-wall camera cadence in controller frames. The later assignment below overrides this value.
FLOATING_WALL_DETECT_EVERY  = 6
# Uses every nth lidar ray during mapping; a value of one keeps every ray.
LIDAR_SUBSAMPLE             = 1


# Sets the lidar frontier range in metres for the related geometry or distance check.
LIDAR_FRONTIER_RANGE        = 6.0
# Amount subtracted from a cell's log-odds when a lidar ray confirms that the cell is free.
LIDAR_FREE_DECREMENT        = 1.50
# Amount added to a cell's log-odds when a lidar return marks that cell as occupied.
LIDAR_WALL_INCREMENT        = 2.00
# Number of frames that mapping waits after a turn so moving scans do not smear walls.
LIDAR_POST_TURN_SETTLE      = 5


# Selects the depth-first lighthouse exploration strategy when enabled.
USE_DFS_LIGHTHOUSE_EXPLORATION = False


# Enables pose-graph SLAM and loop-closure correction when true.
USE_POSE_GRAPH      = True
# Sets the node spacing metres in metres for the related geometry or distance check.
NODE_SPACING_M      = 0.20
# Sets the loop radius metres in metres for the related geometry or distance check.
LOOP_RADIUS_M       = 0.50
# Minimum node index separation used to prevent a scan from matching its recent neighbors.
LOOP_TIME_GAP       = 30
# Sets the cutoff used by the loop fitness thresh decision.
LOOP_FITNESS_THRESH = 0.50


# Stores the driven wheel radius in metres for encoder odometry.
WHEEL_RADIUS = 0.0425
# Stores the distance in metres between the left and right wheel tracks.
WHEEL_TREAD  = 0.165

# Stores half the protected robot length in metres for collision checks.
ROBOT_HALF_L = 0.120
# Stores half the protected robot width in metres for collision checks.
ROBOT_HALF_W = 0.150
# Stores the complete protected robot length in metres.
ROBOT_LENGTH = 0.24
# Stores the complete protected robot width in metres.
ROBOT_WIDTH  = 0.30
# Stores the robot height in metres for deciding which overhead obstacles are passable.
ROBOT_HEIGHT = 0.25

# Legacy pillar radius in metres retained for code paths that use the older geometry name.
PILLAR_RADIUS = 0.02

# Sets the pillar height metres in metres for the related geometry or distance check.
PILLAR_HEIGHT_M = 0.30
# Sets the pillar radius metres in metres for the related geometry or distance check.
PILLAR_RADIUS_M = 0.10
# Sets the pillar diameter metres in metres for the related geometry or distance check.
PILLAR_DIAMETER_M = 1.0 * PILLAR_RADIUS_M

# Sets the pillar arrive distance in metres for the related geometry or distance check.
PILLAR_ARRIVE_DIST = ROBOT_HALF_L + PILLAR_RADIUS_M

# Sets the blue touch sensor distance in metres for the related geometry or distance check.
BLUE_TOUCH_SENSOR_DIST = 0.05
# Sets the blue touch timeout duration in seconds before the related action changes or stops.
BLUE_TOUCH_TIMEOUT     = 6.0


# Sets the pillar visibility minimum fraction value used by the related controller logic.
PILLAR_VISIBILITY_MIN_FRAC   = 0.55
# Sets the pillar occlusion maximum distance in metres for the related geometry or distance check.
PILLAR_OCCLUSION_MAX_DIST    = 3.5
# Sets the pillar occlusion minimum distance in metres for the related geometry or distance check.
PILLAR_OCCLUSION_MIN_DIST    = 0.35


# Sets the pillar occlusion stamp fraction value used by the related controller logic.
PILLAR_OCCLUSION_STAMP_FRAC  = 0.35
# Sets the pillar occlusion minimum frames as a number of controller frames or update cycles.
PILLAR_OCCLUSION_MIN_FRAMES  = 3
# Camera height above the floor in metres for ground-plane projection.
CAM_H        = 0.188
# Expected wall-top height in metres for depth and occlusion calculations.
WALL_TOP_H   = 0.75

# Required distance in metres from the robot center to an obstacle in front.
SAFE_FRONT   = ROBOT_HALF_L + 0.05
# Required distance in metres from the robot center to an obstacle at either side.
SAFE_SIDE    = ROBOT_HALF_W + 0.05
# Required diagonal clearance in metres when the chassis sweeps through a turn.
SAFE_DIAG    = ROBOT_HALF_W + 0.04


# Sets the forward margin in metres for the related geometry or distance check.
FWD_MARGIN   = 0.08
# Front clearance in metres at which forward motion must stop immediately.
HARD_STOP    = 0.03
# Sets the turn margin in metres for the related geometry or distance check.
TURN_MARGIN  = 0.03



# Sets the cutoff used by the infrared corner warn limit decision.
IR_CORNER_WARN_LIMIT = 0.08
# Infrared corner distance in metres that forces an immediate stop during a turn.
IR_CORNER_HARD_STOP  = 0.03



# Sets the x-axis component of the lidar offset in metres relative to the robot origin.
LIDAR_OFFSET_X   =  0.020
# Sets the y-axis component of the lidar offset in metres relative to the robot origin.
LIDAR_OFFSET_Y   =  0.000
# Sets the z-axis component of the lidar offset in metres relative to the robot origin.
LIDAR_OFFSET_Z   =  0.180
# Sets the yaw angle of the lidar offset in radians.
LIDAR_OFFSET_YAW =  0.000

# Maps each infrared sensor name to its x, y, z, and yaw mounting offsets in the robot frame.
IR_OFFSETS = {
    'fl': ( 0.105,  0.080, 0.055,  math.pi / 4),
    'fr': ( 0.105, -0.080, 0.055, -math.pi / 4),
    'rl': (-0.105,  0.080, 0.055,  3 * math.pi / 4),
    'rr': (-0.105, -0.080, 0.055, -3 * math.pi / 4),
}

# Sets the x-axis component of the camera offset in metres relative to the robot origin.
CAMERA_OFFSET_X   =  0.000
# Sets the y-axis component of the camera offset in metres relative to the robot origin.
CAMERA_OFFSET_Y   =  0.011
# Sets the z-axis component of the camera offset in metres relative to the robot origin.
CAMERA_OFFSET_Z   =  0.199
# Sets the pitch angle of the camera offset in radians.
CAMERA_OFFSET_PITCH = 0.0
# Sets the yaw angle of the camera offset in radians.
CAMERA_OFFSET_YAW   = 0.0
# Sets the x-axis component of the depth offset in metres relative to the robot origin.
DEPTH_OFFSET_X    =  0.000
# Sets the y-axis component of the depth offset in metres relative to the robot origin.
DEPTH_OFFSET_Y    =  0.037
# Sets the z-axis component of the depth offset in metres relative to the robot origin.
DEPTH_OFFSET_Z    =  0.199
# Sets the maximum ground distance in metres for the related geometry or distance check.
MAX_GROUND_DIST     = 3.0



# Sets the inclusive lower hue, saturation, and value limits for green floor pixels.
GREEN_HSV_LOW   = (38, 80, 50)
# Sets the inclusive upper hue, saturation, and value limits for green floor pixels.
GREEN_HSV_HIGH  = (90, 255, 255)



# Sets the green minimum stamp distance in metres for the related geometry or distance check.
GREEN_MIN_STAMP_DIST  = 0.20
# Sets the green maximum stamp distance in metres for the related geometry or distance check.
GREEN_MAX_STAMP_DIST  = 1.5



# Top edge of the usable green-floor image band as a fraction of image height.
GREEN_ROW_MIN_FRAC   = 0.55
# Bottom edge of the usable green-floor image band as a fraction of image height.
GREEN_ROW_MAX_FRAC   = 0.92

# Adds the green depth bias correction to compensate for a consistent measurement offset.
GREEN_DEPTH_BIAS     = 0.0



# Sets the cutoff used by the green vote threshold decision.
GREEN_VOTE_THRESHOLD  = 3
# Sets the green vote decay as a number of controller frames or update cycles.
GREEN_VOTE_DECAY      = 60


# Radius in map cells used when a confirmed green-floor point is written to the map.
GREEN_STAMP_RADIUS    = 1


# Sets the sample or item count used for green subsample stride.
GREEN_SUBSAMPLE_STRIDE = 3



# Sets the pillar detect range in metres for the related geometry or distance check.
PILLAR_DETECT_RANGE = 3.5
# Smallest camera contour area in pixels accepted as a possible colored pillar.
PILLAR_MIN_CONTOUR_AREA = 35
# Sets the pillar confirm distance in metres for the related geometry or distance check.
PILLAR_CONFIRM_DIST = 2.5
# Sets the sample or item count used for pillar navigation minimum buf.
PILLAR_NAV_MIN_BUF  = 5
# Sets the close approach distance in metres for the related geometry or distance check.
CLOSE_APPROACH_DIST = 0.50
# Sets the close confirm distance in metres for the related geometry or distance check.
CLOSE_CONFIRM_DIST  = ROBOT_HALF_L + PILLAR_RADIUS_M + 0.01
# Sets the green obstacle confirm distance in metres for the related geometry or distance check.
GREEN_OBSTACLE_CONFIRM_DIST = 1.2
# Half-size in metres of the protected square placed around a confirmed green obstacle.
GREEN_HALF = 0.25

# Sets the cutoff used by the narrow gap threshold decision.
NARROW_GAP_THRESHOLD = 0.90
# Enables gap closing when set to true.
ENABLE_GAP_CLOSING = False


# Sets the pillar tiny minimum pixels value used by the related controller logic.
PILLAR_TINY_MIN_PX   = 4
# Sets the pillar vlock maximum pixels value used by the related controller logic.
PILLAR_VLOCK_MAX_PX  = 2000
# Sets the visual lock timeout duration in seconds before the related action changes or stops.
VISUAL_LOCK_TIMEOUT  = 4.0
# Sets the visual lock cooldown duration in seconds before the related action changes or stops.
VISUAL_LOCK_COOLDOWN = 10.0


# Enables junction scan when set to true.
ENABLE_JUNCTION_SCAN = False


# Enables narrow gap walls when set to true.
ENABLE_NARROW_GAP_WALLS      = True


# Enables camera debug when set to true.
SHOW_CAMERA_DEBUG = False


# Moves frontier goals toward the safest local corridor center when enabled.
FRONTIER_CENTER_SNAP = True


# Names the mission state used while controller services are starting.
STATE_STARTUP    = 'STARTUP'
# Names the mission state used for the initial surroundings scan.
STATE_PRESCAN    = 'PRESCAN'

# Names the mission state used while mapping unknown areas.
STATE_EXPLORE    = 'EXPLORE'
# Names the mission state used while travelling to the blue pillar.
STATE_NAV_BLUE   = 'NAV_BLUE'
# Names the mission state used while travelling to the yellow pillar.
STATE_NAV_YELLOW = 'NAV_YELLOW'
# Names the mission state used when a required target still needs to be found.
STATE_SEARCH_TARGET = 'SEARCH_TARGET'
# Names the mission state used to face and classify a possible pillar.
STATE_LOOK_AT_CANDIDATE = 'LOOK_AT_CANDIDATE'
# Names the final mission state after all required work is complete.
STATE_DONE       = 'DONE'


# Sets the path draw width in metres for the related geometry or distance check.
PATH_DRAW_WIDTH = 0.16
# Distance in metres at which the robot considers the current path goal reached.
PATH_ARRIVE_TOL = 0.08
# Distance in metres that allows the path follower to skip an already passed waypoint.
PATH_SKIP_TOL   = 0.15
# Sets the path lookahead in metres for the related geometry or distance check.
PATH_LOOKAHEAD  = 0.40

# Lists obstacle inflation kernel sizes from safest to least restrictive for repeated path attempts.
PLANNER_KERNELS = [11, 9, 7, 5, 3, 1, 0]

# Sets how strongly A-star hug contributes to the combined score.
ASTAR_HUG_WEIGHT       = 45.0
# Sets how strongly A-star narrow contributes to the combined score.
ASTAR_NARROW_WEIGHT    = 2.0
# Sets how strongly A-star turn contributes to the combined score.
ASTAR_TURN_WEIGHT      = 2.0
# Standard deviation in metres of the smooth wall-proximity cost used by A-star.
ASTAR_GAUSSIAN_STD     = 0.15
# Sets how strongly A-star centering contributes to the combined score.
ASTAR_CENTERING_WEIGHT = 150.0
# Sets how strongly A-star visit contributes to the combined score.
ASTAR_VISIT_WEIGHT     = 0.5
# Sets how strongly A-star trail contributes to the combined score.
ASTAR_TRAIL_WEIGHT     = 1.0
# Sets how strongly A-star geom contributes to the combined score.
ASTAR_GEOM_WEIGHT      = 3.0
# Sets how strongly A-star corner contributes to the combined score.
ASTAR_CORNER_WEIGHT    = 5.0
# Weight given to corridor width when ranking exploration frontiers.
FRONTIER_WIDTH_WEIGHT  = 1.8
# Sets how strongly frontier novelty contributes to the combined score.
FRONTIER_NOVELTY_WEIGHT = 20.0
# Sets how strongly frontier distance contributes to the combined score.
FRONTIER_DISTANCE_WEIGHT = 20.0


# Sets the control maximum forward speed limit used when producing motion commands.
CTRL_MAX_VX         = 0.60
# Sets the control maximum angular speed limit used when producing motion commands.
CTRL_MAX_OM         = 4.0
# Sets the control maximum wheel rad limit used when producing motion commands.
CTRL_MAX_WHEEL_RAD  = 10.0



# Sets the approach maximum angular speed limit used when producing motion commands.
APPROACH_MAX_OM     = 3.5
# Sets the approach tight angular speed limit used when producing motion commands.
APPROACH_TIGHT_OM   = 1.2
# Sets the proportional response strength for approach angular speed.
APPROACH_OM_GAIN    = 4.0
# Sets the proportional response strength for approach steer.
APPROACH_STEER_GAIN = 2.5

# Sets the follow speed base limit used when producing motion commands.
FOLLOW_SPEED_BASE      = 0.40
# Sets the follow speed tight limit used when producing motion commands.
FOLLOW_SPEED_TIGHT     = 0.28
# Sets the follow lookahead base in metres for the related geometry or distance check.
FOLLOW_LOOKAHEAD_BASE  = 0.20
# Sets the follow lookahead tight in metres for the related geometry or distance check.
FOLLOW_LOOKAHEAD_TIGHT = 0.15
# Sets the proportional response strength for follow angular speed.
FOLLOW_OM_GAIN         = 2.6

# Sets the recovery stuck ticks as a number of controller frames or update cycles.
RECOVERY_STUCK_TICKS = 80
# Sets the recovery stuck distance in metres for the related geometry or distance check.
RECOVERY_STUCK_DIST  = 0.01
# Sets the recovery timeout l2 duration in seconds before the related action changes or stops.
RECOVERY_TIMEOUT_L2  = 4.0
# Sets the recovery escape distance in metres for the related geometry or distance check.
RECOVERY_ESCAPE_DIST = 0.30


# Number of failed navigation grace periods allowed before full stuck recovery starts.
NAV_LOOP_BREAKER_ESCALATE    = 3
# Sets the navigation recovery suppress ticks as a number of controller frames or update cycles.
NAV_RECOVERY_SUPPRESS_TICKS  = 120


# Sets the visual range in metres for the related geometry or distance check.
VISUAL_RANGE           = 0.65
# Sets the explore hard timeout duration in seconds before the related action changes or stops.
EXPLORE_HARD_TIMEOUT   = 180.0
# Sets the explore cluster r in occupancy-grid cells.
EXPLORE_CLUSTER_R      = 15
# Sets the explore replan every as a number of controller frames or update cycles.
EXPLORE_REPLAN_EVERY   = 100
# Sets the cutoff used by the breadth-first search strike threshold decision.
BFS_STRIKE_THRESHOLD   = 3
# Sets the junction reach distance in metres for the related geometry or distance check.
JUNCTION_REACH_DIST    = 0.45
# Sets the junction cluster distance in metres for the related geometry or distance check.
JUNCTION_CLUSTER_DIST  = 1.50


# Sets the coordinate record distance in metres for the related geometry or distance check.
COORD_RECORD_DIST = 0.20
# Maximum number of recorded position samples retained in the rolling history.
COORD_HIST_MAX    = 500


# Sets the inclusive lower hue, saturation, and value limits for wood-like pixels.
WOOD_HSV_LOW               = (8, 60, 70)
# Sets the inclusive upper hue, saturation, and value limits for wood-like pixels.
WOOD_HSV_HIGH              = (28, 255, 255)
# Adds the floating depth bias correction to compensate for a consistent measurement offset.
FLOATING_DEPTH_BIAS        = 0.0
# Sets the height margin below in metres for the related geometry or distance check.
Z_MARGIN_BELOW             = 0.04



# Highest world height in metres considered by the broad wood-depth detector.
Z_CEILING                  = 0.66
# Sets the maximum detect distance in metres for the related geometry or distance check.
MAX_DETECT_DIST            = 3.2
# Effective number of controller frames between floating-wall detections; this overrides the earlier value.
FLOATING_WALL_DETECT_EVERY = 3


# Nearest depth in metres included in the emergency floating-wall danger band.
FLOATING_WALL_DANGER_MIN   = 0.03
# Farthest depth in metres included in the emergency floating-wall danger band.
FLOATING_WALL_DANGER_MAX   = 0.40
# Sets the sample or item count used for floating wall danger count.
FLOATING_WALL_DANGER_COUNT = 35
# Log-odds added to map cells when a floating wall observation is confirmed.
FLOATING_WALL_LOG_INCR     = 1.5
# Sets the floating wall radius cells in occupancy-grid cells.
FLOATING_WALL_RADIUS_CELLS = 2

# Rejects floating-wall points lower than this height in metres as floor noise.
FLOATING_Z_FLOOR_GUARD     = 0.015


# Highest obstacle point in metres treated as physically blocking the robot.
FLOATING_Z_CEILING         = 0.26
# Higher temporary ceiling in metres used to collect the full shape of a blocking wall.
FLOATING_Z_CEILING_RELAXED = 0.80
# Sets the cutoff used by the floating height blocking threshold decision.
FLOATING_Z_BLOCKING_THRESHOLD = 0.21
# Sets the floating maximum distance in metres for the related geometry or distance check.
FLOATING_MAX_DIST          = 2.0
# Sets the sample or item count used for floating vertical k pixels.
FLOATING_VERTICAL_K_PIXELS = 6
# Sets the sample or item count used for floating wall minimum points.
FLOATING_WALL_MIN_POINTS   = 5
# Sets the floating wall connect gap in occupancy-grid cells.
FLOATING_WALL_CONNECT_GAP  = 5
# Half-angle in radians of the camera cone accepted for floating-wall projection.
FLOATING_FOV_GUARD         = 0.7
# Sets the floating wall decay frames as a number of controller frames or update cycles.
FLOATING_WALL_DECAY_FRAMES = 60



# Sets the cutoff used by the floating vote threshold decision.
FLOATING_VOTE_THRESHOLD    = 4
# Sets the floating vote decay frames as a number of controller frames or update cycles.
FLOATING_VOTE_DECAY_FRAMES = 50
# Sets the floating minimum wall width metres in metres for the related geometry or distance check.
FLOATING_MIN_WALL_WIDTH_M  = 0.08



# Number of map cells added around lidar walls when rejecting duplicate camera detections.
FLOATING_LIDAR_STRIP_MARGIN_CELLS = 10
# Sets the floating wall minimum distance metres in metres for the related geometry or distance check.
FLOATING_WALL_MIN_DIST_M = 0.10


# Sets the floating noise minimum area cells in occupancy-grid cells.
FLOATING_NOISE_MIN_AREA_CELLS = 18
# Sets the cutoff used by the floating noise blob thresh decision.
FLOATING_NOISE_BLOB_THRESH    = 40
# Minimum length-to-width ratio required for a small blob to look like a wall segment.
FLOATING_NOISE_MIN_ASPECT     = 1.8




# Minimum open lidar distance in metres needed to infer an obstacle below the lidar plane.
FLOATING_STUCK_LIDAR_OPEN     = 0.30
# Sets the inferred floating width in metres for the related geometry or distance check.
INFERRED_FLOATING_WIDTH       = ROBOT_WIDTH + 0.12
# Sets the inferred floating ahead in metres for the related geometry or distance check.
INFERRED_FLOATING_AHEAD       = 0.14
# Sets the sample or item count used for inferred floating n points.
INFERRED_FLOATING_N_POINTS    = 15
# Sets the inferred floating radius cells in occupancy-grid cells.
INFERRED_FLOATING_RADIUS_CELLS = 3
# Sets the inferred floating forward reach in metres for the related geometry or distance check.
INFERRED_FLOATING_FORWARD_REACH = 0.28
# Sets the sample or item count used for inferred floating forward steps.
INFERRED_FLOATING_FORWARD_STEPS = 5


# Sets the sample or item count used for floating weak wood pixels.
FLOATING_WEAK_WOOD_PIXELS          = 80
# Sets the sample or item count used for floating depth only minimum points.
FLOATING_DEPTH_ONLY_MIN_POINTS     = 12
# Fraction of danger-band depth pixels needed for depth evidence to contribute to fusion.
FLOATING_FUSION_DEPTH_DANGER_FRAC  = 0.25
# Sets how strongly floating fusion depth contributes to the combined score.
FLOATING_FUSION_DEPTH_WEIGHT       = 1.0
# Sets how strongly floating fusion wood contributes to the combined score.
FLOATING_FUSION_WOOD_WEIGHT        = 0.9
# Sets how strongly floating fusion stuck contributes to the combined score.
FLOATING_FUSION_STUCK_WEIGHT       = 1.2
# Sets how strongly floating fusion lidar open contributes to the combined score.
FLOATING_FUSION_LIDAR_OPEN_WEIGHT  = 0.6
# Sets how strongly floating fusion tactile contributes to the combined score.
FLOATING_FUSION_TACTILE_WEIGHT     = 1.0
# Combined evidence score required to stamp a camera-observed floating wall.
FLOATING_FUSION_CAMERA_STAMP_SCORE = 1.85
# Combined evidence score required to stamp a wall inferred from failed motion.
FLOATING_FUSION_INFER_STAMP_SCORE  = 2.0
# Sets how strongly floating fusion pillar contributes to the combined score.
FLOATING_FUSION_PILLAR_WEIGHT      = 1.1



# Sets the floating wall confirm distance metres in metres for the related geometry or distance check.
FLOATING_WALL_CONFIRM_DIST_M       = 0.55
# Sets the floating wall pending maximum frames as a number of controller frames or update cycles.
FLOATING_WALL_PENDING_MAX_FRAMES   = 120

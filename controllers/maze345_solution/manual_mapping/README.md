# Manual Mapping Controller

This folder contains a Webots controller for exploring a maze, building an occupancy map, locating colored pillars, detecting obstacles at different heights, and navigating between mission targets. The controller uses the standard Webots `Robot` API and does not require supervisor access.

## How the controller works

The controller starts in `manual_mapping.py`. It creates the sensor, localization, perception, SLAM, planning, motion, recovery, persistence, state-machine, and visualization services. These services share the same robot pose and map.

During each Webots step, the controller performs the following work:

1. Read wheel and inertial data to update the robot pose.
2. Update the explored trail and occupancy map at their configured rates.
3. Process camera, depth, and lidar data for pillars and floating walls.
4. Record position history periodically.
5. Run recovery if the robot is stuck; otherwise run the current mission state.
6. Refresh enabled diagnostic displays at a reduced rate.

The work is spread across several frame intervals to keep motion responsive and reduce unnecessary CPU use.

## File descriptions

### `manual_mapping.py`

The main controller entry point. It creates the Webots `Robot`, initializes every controller service, connects shared dependencies, and runs the simulation loop. It also filters console messages and writes selected state and coordinate messages to `debug_log.txt`.

### `config.py`

Contains all tuning values used by the controller. This includes map size, sensor mounting positions, robot dimensions, color limits, path-planning weights, speed limits, recovery thresholds, and floating-wall detection settings. Each variable has a nearby explanation of its purpose and unit.

### `sensors.py`

Defines `SensorSuite`, which initializes and provides access to the Webots motors, wheel encoders, inertial unit, lidar, infrared sensors, RGB camera, and depth camera. It caches converted camera and depth frames during a simulation step and calculates obstacle clearance around the chassis.

### `localization.py`

Defines `Localizer`, which estimates the robot's world position and heading. It combines wheel encoder movement with inertial heading and accepts reliable corrections from scan matching. Other modules use this pose for mapping, planning, and object projection.

### `slam.py`

Contains the main mapping implementation. `OccupancyMap` maintains free, unknown, and occupied cells from lidar observations. `PoseGraphSLAM` adds pose history, scan matching, and loop-closure correction. The `make_slam` function chooses the configured SLAM implementation.

### `pose_graph.py`

Defines the pose-graph data structures. Pose nodes store robot positions and scans, while edges store odometry or loop-closure constraints between nodes. The graph provides the data required by the optimizer to correct accumulated drift.

### `scan_matcher.py`

Aligns two lidar scans with point-to-point iterative closest point matching. It converts ranges into points, repeatedly estimates the relative position and heading, and returns a fitness score. Weak matches can be rejected before they affect localization or the pose graph.

### `loop_closure.py`

Looks for older pose-graph nodes near the robot's current position. It uses scan matching to verify whether two visits represent the same place. A successful match adds a loop-closure constraint that can correct the shape of the map.

### `graph_optimizer.py`

Optimizes the complete pose graph with a Gauss-Newton solver. It spreads position and heading error across connected poses while keeping the first pose fixed as the map origin. The corrected poses reduce drift after a loop closure is accepted.

### `perception.py`

Defines `PillarDetector`, which handles the main environment perception tasks. It detects blue and yellow pillars, maps green floor areas, finds round lidar candidates, and detects walls that may sit above or below the lidar scan plane. Camera and depth observations are projected from the robot frame into world and map coordinates.

### `floating_fusion.py`

Combines evidence for floating walls from camera color, depth, lidar clearance, pillar observations, contact-like behavior, and failed movement. It confirms repeated evidence before stamping obstacles into the map and removes stale or weak detections to reduce false walls.

### `planner.py`

Defines `Planner`, which searches the occupancy map for safe routes. It inflates obstacles to account for robot size, ranks exploration frontiers, checks path validity, smooths paths, records visited trails, and selects alternative clearance kernels when a preferred path is unavailable.

### `motion.py`

Defines `MotionController`, which converts paths and short movement requests into wheel speeds. It follows waypoints, limits speed and turning, checks front, rear, side, and corner clearance, and reports poor progress to recovery. It also handles close approaches to mission targets.

### `recovery.py`

Defines `RecoveryManager`, which detects when the robot is not moving as expected. Recovery begins with local turns toward open space and can escalate to larger escape actions or a reachable frontier. Clearance checks protect the full chassis during these movements.

### `state_machine.py`

Defines `MissionStateMachine`, which controls the mission sequence. It manages startup, scanning, exploration, visual confirmation, navigation to blue and yellow pillars, target searching, and completion. It can briefly interrupt normal driving to inspect uncertain pillar or floating-wall observations.

### `persistence.py`

Defines `PersistenceManager`, which records sampled robot positions and confirmed target coordinates. The history length is limited to control memory use. It can save the collected mission history to `coordinate_history.json` when requested.

### `visualization.py`

Defines `Visualizer`, which draws the live occupancy map, robot pose, heading, planned path, visited trail, detected pillars, and floating walls. Display updates are throttled so visualization has a smaller effect on controller performance.

## Main dependencies

The controller expects the Webots Python controller package and uses NumPy and OpenCV for numeric processing, image analysis, map operations, and visualization. Some pose-graph operations may also use SciPy when available.

## Running in Webots

Set the robot controller to `manual_mapping` and start the simulation. Webots loads `manual_mapping.py`, which creates the standard `Robot` instance and begins the control loop. Behavior and optional display windows can be adjusted in `config.py`.

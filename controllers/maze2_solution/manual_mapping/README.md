# Python files in manual_mapping_maze2

This folder contains the robot controller used for maze 2. It covers sensor reading, mapping, route planning, pillar detection, movement, and recovery. `manual_mapping.py` is the file that starts everything.

## config.py

This file holds the settings shared by the other modules. They include robot size, wheel speeds, sensor positions, map measurements, planner limits, colour detection ranges, recovery timings, and pose-graph options. The maze 2 values are tuned separately from the other maze folders.

## floating_fusion.py

The `FloatingWallFusion` class joins several clues to decide whether an obstacle is a floating wall. It checks depth readings, wood-like colour, possible pillar blocking, and stalled movement. Evidence is built up over several readings before map points are accepted, which helps prevent single bad readings from leaving false walls on the map.

## graph_optimizer.py

This file improves the positions in a pose graph with Gauss-Newton optimisation. It calculates the error and Jacobians for each graph edge, keeps the starting pose fixed, and adjusts the later poses. `optimize_pose_graph` is the public function called after a useful loop closure is found.

## localization.py

The `Localizer` estimates the robot's x position, y position, and heading. Wheel encoders supply movement, while the IMU helps correct rotation drift. The map-correction method can compare current lidar readings with nearby occupied map cells and apply a limited correction when the match is trustworthy.

## loop_closure.py

The `LoopClosureDetector` checks whether the robot has returned to an older part of the maze. It selects nearby older pose nodes and uses the lidar scan matcher to compare them with the current scan. A strong match is added to the graph as a loop-closure edge so accumulated position error can be corrected.

## manual_mapping.py

This is the main controller script. It creates the Webots robot and all of the helper objects, then runs the step-by-step control loop. During each step it gathers sensor data, updates localization and mapping, runs perception, lets the mission state machine choose an action, drives the motors, updates the display, and records run data. `custom_print` is a small helper for controlling printed messages.

## motion.py

The `MotionController` follows paths produced by the planner. It turns toward path points, drives at a suitable speed, and sends values to both wheel motors. Safety checks use lidar, map obstacles, chassis clearance, and floating-wall evidence. The controller can stop an unsafe approach and can look for another nearby frontier when a route becomes unsuitable.

## perception.py

The `PillarDetector` reads colour and depth camera images to find green ground and coloured pillars. It projects useful pixels into world coordinates and also checks lidar shapes that may be pillars missing from the map. Floating-wall detection is kept here too, since it needs camera and lidar information from the same scene.

## persistence.py

The `PersistenceManager` records the path followed by the robot. It creates the coordinate-history output, adds new positions, and saves the history in the required format. Unlike the versions in the other maze folders, this version concentrates on coordinate history and does not contain a `save_map` method.

## planner.py

The `Planner` creates safe routes across the occupancy grid. It builds a cost map that gives extra space to walls, blocks narrow gaps, finds unexplored frontiers, and runs A-star search. It can reuse a path while it remains valid, smooth clear sections, reject unsafe corners, check direct line of sight, and stamp the travelled trail onto the map.

## pose_graph.py

`PoseNode` stores a recorded robot pose and scan. `PoseEdge` stores the movement constraint between two nodes. The `PoseGraph` class manages nodes, odometry edges, loop-closure edges, and corrected poses. Helper functions create suitable information matrices so reliable constraints have more effect during optimisation.

## recovery.py

The `RecoveryManager` detects poor movement and tries to free the robot. It checks whether rotations are safe, measures lidar clearance in chosen directions, and compares expected movement with real movement. Maze 2 includes extra navigation updates and specialised stuck handling. Recovery actions can reverse, turn, nudge forward, or respond to a possible floating wall depending on the available space.

## scan_matcher.py

This file matches lidar scans using point-to-point ICP. It converts scan ranges to points, transforms the points, finds the nearest reference point, and improves the pose estimate over several rounds. It normally uses SciPy's fast nearest-neighbour search and includes a brute-force fallback. `match_scans` returns the estimated movement and a score showing how well the scans agree.

## sensors.py

The `SensorSuite` collects all Webots sensor access in one place. It enables the cameras, lidar, encoders, and IMU, then returns their data in forms used by the rest of the controller. Its distance helpers estimate clearance around the robot body and corners, which is useful for close movement and recovery.

## slam.py

This is the mapping module. `OccupancyMap` ray-traces lidar readings into free and occupied grid cells, preserves green areas and floating walls, clears stale noise, handles narrow gaps, and extracts useful wall shapes. The maze 2 version includes extra wall thinning and geometric map cleanup. `PoseGraphSLAM` adds scan nodes, loop closure, graph optimisation, and full map rebuilding. `make_slam` selects which mapping class to use.

## state_machine.py

The `MissionStateMachine` decides what the robot should do next. It manages exploration, blue and yellow pillar targets, touching and confirming pillars, junction scans, and route changes. The maze 2 version has extra checks for safe explored paths, a home target, a dedicated blue-target recovery routine, final blue and yellow navigation methods, green exclusion zones, and selection of uncovered camera viewpoints.

## visualization.py

The `Visualizer` creates the live OpenCV map window. It draws the occupancy grid together with the robot pose, route, trail, targets, detected pillars, junctions, and other state information. It is used to inspect progress and understand why the controller selected its current movement.

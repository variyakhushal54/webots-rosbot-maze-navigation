# Manual Mapping Controller

This directory contains a Webots robot controller for mapping and navigating a maze. The controller reads the robot sensors, estimates its position, builds an occupancy map, detects pillars and special obstacles, plans safe paths, and manages the mission from startup to completion.

The controller uses the standard Webots `Robot` API. It does not require Supervisor permissions.

## Main file

### `manual_mapping.py`

This is the controller entry point. It creates the robot, sensors, localization, mapping, perception, planning, movement, recovery, persistence, and visualization objects. Its main loop reads sensor data, updates the map, runs the mission state machine, and displays the current result.

## Configuration

### `config.py`

This file contains the constants used by the whole controller. It includes map dimensions, robot geometry, sensor positions, detection limits, planner weights, movement speeds, recovery settings, and visualization switches. Every constant has a nearby comment explaining its purpose and unit.

Change controller behaviour here before changing values inside the other modules.

## Sensors and perception

### `sensors.py`

Sets up the Webots devices and provides camera, depth, lidar, wheel, keyboard, and infrared readings in forms that the other modules can use.

### `perception.py`

Processes camera, depth, and lidar data. It detects coloured pillars, green ground areas, floating walls, and possible unmapped lidar pillars. Image detections are converted into positions that can be placed on the map.

### `floating_fusion.py`

Combines evidence from depth, wood colour, lidar visibility, pillar detections, and stuck events. It confirms likely floating walls while rejecting weak or noisy detections.

## Position and mapping

### `localization.py`

Tracks the robot position and heading from wheel movement and available sensor corrections. The estimated pose is used by mapping, perception, planning, and movement.

### `slam.py`

Builds the occupancy map while the robot moves. Lidar rays mark free cells and walls, special detections add other obstacles, and pose corrections keep older map data aligned.

### `pose_graph.py`

Stores pose nodes and the measured connections between them. Connections may represent normal robot movement or a loop closure with an older location.

### `scan_matcher.py`

Aligns two lidar scans with an iterative closest-point calculation. It returns the estimated transform, a fitness score, and information used by pose-graph optimization.

### `loop_closure.py`

Searches for older poses near the robot's current position. It compares their scans and adds a loop edge when the match is reliable enough.

### `graph_optimizer.py`

Adjusts pose-graph nodes so movement and loop-closure measurements agree. Corrected poses reduce accumulated position drift and improve map alignment.

## Planning and movement

### `planner.py`

Builds cost maps and searches for safe routes with A-star. It considers walls, robot clearance, narrow gaps, unknown areas, visited paths, and route smoothness. It also selects exploration frontiers and checks whether an existing path remains valid.

### `motion.py`

Converts paths and navigation targets into wheel commands. It controls speed and steering while checking obstacle clearance, corners, tight corridors, and close approaches.

### `recovery.py`

Detects when the robot is stuck and runs staged escape actions. It checks whether rotations and reverse movements are safe, then returns control to normal navigation after the robot moves again.

### `state_machine.py`

Controls the mission sequence. It manages startup, initial scanning, exploration, target searching, pillar confirmation, navigation, recovery coordination, and mission completion.

## Output and debugging

### `visualization.py`

Draws the occupancy map, robot pose, planned path, detected pillars, floating walls, and other debugging markers. Display windows can be enabled or disabled in `config.py`.

### `persistence.py`

Records the robot path and confirmed pillar coordinates. It writes data in a form that can be inspected or reused during the controller run.


## How the modules work together

1. `manual_mapping.py` creates the controller components.
2. `sensors.py` reads the Webots devices.
3. `localization.py` estimates the current robot pose.
4. `slam.py` and the pose-graph modules update the map.
5. `perception.py` and `floating_fusion.py` add visual detections.
6. `state_machine.py` chooses the current mission task.
7. `planner.py` creates a route for that task.
8. `motion.py` follows the route, while `recovery.py` handles failures.
9. `visualization.py` and `persistence.py` show and record the result.

## Requirements

The controller is intended to run inside Webots with Python. The Python environment needs:

- Webots `controller` module
- NumPy
- OpenCV for Python

The Webots robot must provide the device names expected by `sensors.py`.

## Running the controller

Set this directory as the robot controller directory in Webots and use `manual_mapping.py` as the controller entry point. Start the simulation normally. 

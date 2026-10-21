# Autonomous Maze Navigation with the RosBot in Webots

Python controllers that let a simulated **RosBot** explore five unknown mazes, build its own occupancy map, detect coloured target pillars, avoid forbidden floor, and reach the **blue pillar first and then the yellow pillar** — with no prior map and no Supervisor (privileged world) access.

Built as a group project for the *Autonomous Robots* module (M.Sc. Artificial Intelligence for Industrial Applications, OTH Amberg-Weiden), July 2026.

## Highlights

- **5 / 5 mazes solved** in the recorded runs, using only onboard sensors (wheel encoders, IMU, 2D LiDAR, RGB camera, depth camera).
- **Online SLAM-style mapping**: LiDAR ray-traced occupancy grid (1000 × 1000 cells, 0.02 m resolution), scan matching (ICP) and pose-graph optimisation to reduce odometry drift.
- **Vision + depth perception**: HSV colour segmentation for blue/yellow pillars and forbidden green floor, with repeated-frame confirmation to reject false detections.
- **Floating-obstacle fusion**: evidence from depth, colour, LiDAR and stalled motion is accumulated before an obstacle is written to the map.
- **Planning and control**: frontier-based exploration, A* on an inflated cost map, lookahead path following, and a staged stuck-recovery manager.
- **Mission state machine** that enforces the required order: `STARTUP → PRESCAN → EXPLORE → NAV_BLUE → NAV_YELLOW → DONE`.

## Results

Timings from the recorded runs (simulation time, start → target):

| Maze | Start → Blue | Blue → Yellow | Total | Status |
|------|--------------|---------------|-------|--------|
| 1 | 2:51 (171 s) | 1:01 (61 s) | 3:52 (232 s) | Solved |
| 2 | 3:29 (209 s) | 1:44 (104 s) | 5:13 (313 s) | Solved |
| 3 | 2:02 (122 s) | 0:36 (36 s) | 2:38 (158 s) | Solved |
| 4 | 1:23 (83 s) | 1:02 (62 s) | 2:25 (145 s) | Solved |
| 5 | 0:51 (51 s) | 0:27 (27 s) | 1:18 (78 s) | Solved |

Maze 5 was the fastest run; Maze 2 was the longest and has its own dedicated controller with extra route checking and recovery logic.

Detected-map overlays are in [`media/maze_overlays/`](media/maze_overlays/). Recorded runs are hosted on Google Drive (the videos are not stored in this repository to keep it lightweight):

- [Folder with all images and videos](https://drive.google.com/drive/folders/1vXmHBZXA3NbMB7Sq7vobMZSI4hklVjRu)
- Maze 1: [video](https://drive.google.com/file/d/18W7artDSX1nd4FIPeiV9qlfmI08U962r/view?usp=drive_link) · Maze 2: [video](https://drive.google.com/file/d/1gqsUb51Hxj4EvfWU0fjGOyfLZH2SZ4L7/view?usp=drive_link) · Maze 3: [video](https://drive.google.com/file/d/1uXFLbSX5jJYxQFPJsH2q6nwyVwlUqtLG/view?usp=drive_link) · Maze 4: [video](https://drive.google.com/file/d/1ix-YRUBDPCRPTQoX9yAFwMJQvjZ0CVD6/view?usp=drive_link) · Maze 5: [video](https://drive.google.com/file/d/1S8W0fkgOVbnRQEH4Kg8SZocq3Gxc9wV3/view?usp=drive_link)

## Skills demonstrated

- **Robotics:** occupancy-grid mapping, pose estimation (odometry + IMU), scan matching (ICP), pose-graph optimisation, loop closure, path following, stuck detection and recovery
- **Path planning:** A* search, inflated cost maps, frontier-based exploration, path smoothing and validation
- **Computer vision:** HSV colour masking, camera–depth projection, multi-frame confirmation
- **Software engineering:** modular design (one responsibility per file), configuration separated from logic, live visualisation for debugging
- **Tools:** Python, NumPy, OpenCV, SciPy, Webots

## Repository structure

```
.
├── controllers/
│   ├── manual_mapping_maze1_solution/manual_mapping/   # Controller for Maze 1
│   ├── manual_mapping_maze2_solution/manual_mapping/   # Controller for Maze 2 (longest run, dedicated tuning)
│   └── manual_mapping_maze345_solution/manual_mapping/ # Shared controller for Mazes 3, 4 and 5
├── media/maze_overlays/                               # Detected maps for Mazes 1–5
├── report/AUTONOMOUS_REPORT_GROUP_O.pdf               # Full technical report (10 pages)
├── AI_USAGE.md                                        # How AI tools were used in this project
└── README.md
```

Each controller folder contains the same module layout:

| File | Responsibility |
|------|----------------|
| `manual_mapping.py` | Entry point; wires up all subsystems and runs the control loop |
| `config.py` | Map size, speeds, colour thresholds, planner weights, recovery settings |
| `sensors.py` | Access to motors, encoders, IMU, LiDAR, cameras |
| `localization.py` | Pose estimate from wheel motion, IMU yaw and scan correction |
| `slam.py` | Occupancy grid, ray tracing, special map layers, pose-graph SLAM |
| `scan_matcher.py` | Point-to-point ICP between LiDAR scans |
| `pose_graph.py`, `graph_optimizer.py`, `loop_closure.py` | Pose graph, Gauss-Newton optimisation, loop-closure detection |
| `perception.py` | Blue/yellow pillar, green floor and LiDAR pillar detection |
| `floating_fusion.py` | Multi-observation evidence for floating obstacles |
| `planner.py` | Cost map, frontier selection, A* routing, path checks |
| `motion.py` | Turns routes into wheel speeds; local safety checks |
| `recovery.py` | Detects lack of progress and runs staged escape actions |
| `state_machine.py` | Mission logic and state transitions |
| `visualization.py` | Live OpenCV map window |
| `persistence.py` | Coordinate-history and run output |

Per-maze details are in each folder's own `README.md`.

## Running the controllers

These are Webots robot controllers. They need a Webots project containing the provided RosBot robot and maze worlds (not included here).

1. Install [Webots](https://cyberbotics.com/) and the Python dependencies used by the controllers: `numpy`, `opencv-python`, `scipy`. (`g2o` is optional; the graph optimiser falls back to its own implementation.)
2. In your Webots world, set the robot controller to the folder for the maze you want, e.g. `manual_mapping_maze1_solution/manual_mapping`.
3. Run the simulation. The controller prints the mission timers (start → blue, blue → yellow, total) when it finishes.

## Constraints respected

- No use of the Webots Supervisor API; all decisions come from onboard sensors and the robot's own map.
- The provided robot model and world files were not modified.

## Known limitations

- Colour thresholds depend on lighting and camera angle.
- Pillar detection can be delayed when a target is only partly visible.
- Floating-obstacle detection needs enough useful depth evidence.
- Very tight passages can still trigger recovery behaviour.
- Parameters are hand-tuned per maze layout rather than learned.

## Team

- **Khushalkumar Jitendrabhai Variya**
- **Dhruvin Harshadbhai Gopani**

M.Sc. Artificial Intelligence for Industrial Applications, OTH Amberg-Weiden.

## AI usage

See [`AI_USAGE.md`](AI_USAGE.md) for which AI tools were used and for what.

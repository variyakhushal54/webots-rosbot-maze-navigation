# This file starts the robot controller and runs the main mapping loop.
# It connects the sensors, SLAM system, planner, state machine, and visual displays.
# Each loop reads new data, updates the map, chooses an action, and moves the robot.

from controller import Robot
from sensors import SensorSuite
from localization import Localizer
from slam import make_slam, PoseGraphSLAM
from perception import PillarDetector
from planner import Planner
from motion import MotionController
from recovery import RecoveryManager
from state_machine import MissionStateMachine
from visualization import Visualizer
from persistence import PersistenceManager
from floating_fusion import FloatingWallFusion
import config
import builtins
import os
import time



log_file_path = os.path.join(os.path.dirname(__file__), "debug_log.txt")
log_file = open(log_file_path, "w", encoding='utf-8')
original_print = builtins.print

_print_flush_counter = 0

def custom_print(*args, **kwargs):
    global _print_flush_counter
    msg = " ".join(str(a) for a in args)


    msg_lower = msg.lower()
    keep = False
    final_summary_line = ("mission accomplished" in msg_lower or
                          "timer 1" in msg_lower or "timer 2" in msg_lower or
                          "total time" in msg_lower or set(msg.strip()) == {'='})


    if "state" in msg_lower or "gate status" in msg_lower or "switching to" in msg_lower or "transitioning to" in msg_lower or "entering" in msg_lower:
        keep = True

    elif "confirmed" in msg_lower or "locked" in msg_lower or "coordinates" in msg_lower or "pillar" in msg_lower:
        if "green" in msg_lower or "yellow" in msg_lower or "blue" in msg_lower:
            keep = True

    elif ("started" in msg_lower or "starting" in msg_lower or
          "startup" in msg_lower or "devices" in msg_lower or
          "mission start" in msg_lower or final_summary_line):
        keep = True

    if not keep:
        return

    timestamp = f"[{time.strftime('%H:%M:%S')}] "
    log_line = timestamp + msg


    original_print(log_line, **kwargs)


    log_file.write(log_line + "\n")
    _print_flush_counter += 1
    if final_summary_line or _print_flush_counter >= 30:
        log_file.flush()
        _print_flush_counter = 0


builtins.print = custom_print

def main():
    robot    = Robot()
    timestep = int(robot.getBasicTimeStep())

    sensors    = SensorSuite(robot, timestep)

    loc        = Localizer(sensors)
    perception = PillarDetector(sensors)
    slam       = make_slam(config.MAP_SIZE, config.MAP_RES, sensors, loc, perception, robot, timestep)

    perception.loc   = loc
    perception.slam  = slam
    motion           = MotionController(robot, sensors, slam, loc)
    planner          = Planner(slam, loc)
    planner.robot    = robot
    planner.perception = perception
    perception.planner = planner
    floating_fusion  = FloatingWallFusion()
    recovery         = RecoveryManager(motion, sensors, planner, loc)
    recovery.robot   = robot
    recovery.slam    = slam
    recovery.floating_fusion = floating_fusion
    motion.recovery  = recovery
    motion.floating_fusion   = floating_fusion
    history          = PersistenceManager()
    history.loc      = loc
    history.robot    = robot
    history.perception = perception
    history.slam     = slam
    history._init_coord_history()
    state            = MissionStateMachine(perception, planner, motion, recovery, slam, loc, history)
    state.t_start    = robot.getTime()
    recovery.state_machine = state
    recovery.history = history
    motion.state_machine   = state
    motion.planner   = planner
    history.state_machine = state


    viz = Visualizer(slam, loc, planner, perception, state_machine=state)
    history.viz = viz

    cost_viz = None
    temp_viz = None

    frame        = 0
    turn_cooldown = 0

    while robot.step(timestep) != -1:
            frame += 1
            if frame < 5:
                continue
            t = robot.getTime()
            loop_start = time.time()

            loc.update_pose(slam)
            planner.stamp_trail(loc.x, loc.y)
            if state.state != config.STATE_NAV_YELLOW and not getattr(loc, 'slip_active', False):

                if frame % 2 == 0:
                    slam.update_map()




            if frame % 8 == 0:
                perception.detect_pillars_camera(sensors.get_camera_bgr(), *sensors.get_lidar())

            if (frame + 3) % 5 == 0:
                r, w, f = sensors.get_lidar()
                perception.detect_unmapped_lidar_pillars(r, w, f)

            floating_wall_cadence = getattr(config, 'FLOATING_WALL_DETECT_EVERY', 6)
            if (frame + 5) % floating_wall_cadence == 0:
                perception._detect_floating_walls(sensors.get_camera_bgr(), sensors.get_camera_depth())
                floating_fusion.apply(perception, recovery, slam, frame)

            if frame % 30 == 0:
                history._record_position()

            if recovery.stuck_counter >= 3 or recovery.recovery_substate is not None:
                recovery._execute_stuck_recovery()
            else:
                state.run(frame, t)


            if getattr(config, 'ENABLE_MAIN_VIZ', True):
                viz.display()

    log_file.flush()
    log_file.close()

if __name__ == '__main__':
    main()

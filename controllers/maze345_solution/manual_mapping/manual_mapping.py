# This is the entry point for the Webots robot controller. It creates each service,
# connects the shared dependencies, and runs them in a fixed order on every step.
# Expensive camera and mapping work runs at a lower rate than motion control so the
# controller remains responsive without wasting simulation time.
from controller import Robot
from sensors import SensorSuite
from localization import Localizer
from slam import make_slam, OccupancyMap, PoseGraphSLAM
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
    add_timestamp = kwargs.pop('_timestamp', True)
    msg = " ".join(str(a) for a in args)

    
    
    msg_upper = msg.upper()
    keep = False
    if "START" in msg_upper:
        keep = True
    elif "STATE" in msg_upper:
        keep = True
    elif "COORD" in msg_upper and any(c in msg_upper for c in ["GREEN", "YELLOW", "BLUE"]):
        keep = True
    elif any(label in msg_upper for label in ["MISSION ACCOMPLISHED", "TIMER 1", "TIMER 2", "TOTAL TIME"]):
        keep = True
    elif msg and set(msg) == {'='}:
        keep = True

    if not keep:
        return
    timestamp = f"[{time.strftime('%H:%M:%S')}] " if add_timestamp else ""
    log_line = timestamp + msg

    
    original_print(log_line, **kwargs)

    
    log_file.write(log_line + "\n")
    _print_flush_counter += 1
    is_mission_summary = any(label in msg_upper for label in [
        "MISSION ACCOMPLISHED", "TIMER 1", "TIMER 2", "TOTAL TIME"
    ]) or (msg and set(msg) == {'='})
    if _print_flush_counter >= 30 or is_mission_summary:
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
    recovery.state_machine = state
    recovery.history = history
    motion.state_machine   = state
    motion.planner   = planner

    
    viz = Visualizer(slam, loc, planner, perception, state_machine=state)
    history.viz = viz

    
    cost_viz = None
    if getattr(config, 'ENABLE_COSTMAP_VIZ', False):
        try:
            from costmap_visualizer import CostMapVisualizer
            cost_viz = CostMapVisualizer(slam, loc, planner, perception)
            print("[COSTMAP-VIZ] A* Cost Map Visualizer active!")
        except ImportError:
            pass

    
    temp_viz = None
    if getattr(config, 'ENABLE_TEMP_VIZ', True):
        try:
            from temp_floating_wall_visualizer import FloatingWallVisualizer
            temp_viz = FloatingWallVisualizer(slam, perception, loc, sensors, recovery, floating_fusion)
            print("[TEMP-VIZ] Temporary Floating Wall Visualizer active!")
        except ImportError:
            pass
    else:
        print("[TEMP-VIZ] Disabled via ENABLE_TEMP_VIZ=False in config.py")

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

            
            if frame % 4 == 0:
                if getattr(config, 'ENABLE_MAIN_VIZ', True):
                    viz.display()
                if cost_viz is not None:
                    cost_viz.update(frame)
                if temp_viz is not None:
                    temp_viz.display(frame)

    log_file.flush()
    log_file.close()

if __name__ == '__main__':
    main()

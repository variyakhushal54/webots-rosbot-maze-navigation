# SensorSuite owns the Webots devices and presents a consistent interface to them.
# It enables motors, encoders, inertial sensing, lidar, cameras, and range sensors.
# Camera and depth conversions are cached for the current simulation time because
# several perception tasks may request the same frame during one control step.
import config
import math
import numpy as np
import cv2

from controller import Robot, Lidar, Camera, CameraRecognitionObject, GPS, Gyro, Accelerometer, Compass, Motor, PositionSensor

class SensorSuite:
    def __init__(self, robot, timestep):
        self.robot = robot
        self.timestep = timestep

        
        try:
            n_devices = self.robot.getNumberOfDevices()
            print(f"[DEVICES DEBUG] Robot has {n_devices} devices:")
            for i in range(n_devices):
                dev = self.robot.getDeviceByIndex(i)
                print(f"  Device {i}: name='{dev.getName()}', type={dev.getNodeType()}")
        except Exception as e:
            print("[DEVICES DEBUG] Error listing devices:", e)

        self.motors = []
        for n in ['fl_wheel_joint', 'fr_wheel_joint', 'rl_wheel_joint', 'rr_wheel_joint']:
            m = self.robot.getDevice(n)
            if m:
                m.setPosition(float('inf'))
                m.setVelocity(0.0)
                
                try:
                    m.setAcceleration(8.0) 
                except Exception as e:
                    print("[SENSORS] Error setting acceleration limit:", e)
                self.motors.append(m)

        self.encoders = []
        for n in ['front left wheel motor sensor', 'front right wheel motor sensor',
                  'rear left wheel motor sensor', 'rear right wheel motor sensor']:
            s = self.robot.getDevice(n)
            if s:
                s.enable(self.timestep)
                self.encoders.append(s)

        self.imu = self.robot.getDevice('imu inertial_unit')
        if self.imu:
            self.imu.enable(self.timestep)

        self.lidar = self.robot.getDevice('laser')
        if self.lidar:
            self.lidar.enable(self.timestep)

        self.ir = {'front': [], 'rear': []}
        for name in ['fl_range', 'fr_range']:
            s = self.robot.getDevice(name)
            if s:
                s.enable(self.timestep)
                self.ir['front'].append(s)
        for name in ['rl_range', 'rr_range']:
            s = self.robot.getDevice(name)
            if s:
                s.enable(self.timestep)
                self.ir['rear'].append(s)

        self.camera = None
        for cam_name in ['camera rgb', 'Astra', 'astra', 'camera']:
            dev = self.robot.getDevice(cam_name)
            if dev:
                dev.enable(self.timestep)
                self.camera = dev
                self.cam_w, self.cam_h = dev.getWidth(), dev.getHeight()
                self.cam_fov = dev.getFov()
                self.cam_vfov = self.cam_fov * (self.cam_h / self.cam_w)
                if getattr(config, 'SHOW_CAMERA_DEBUG', False):
                    cv2.namedWindow("CAMERA DEBUG", cv2.WINDOW_NORMAL)
                break

        self.rangefinder = None
        for rf_name in ['camera depth', 'range-finder', 'rangefinder', 'astra depth']:
            dev = self.robot.getDevice(rf_name)
            if dev:
                dev.enable(self.timestep)
                self.rangefinder = dev
                self.rf_w = dev.getWidth()
                self.rf_h = dev.getHeight()
                break

    def get_camera_bgr(self):
        """Fetches the raw camera image and converts it from BGRA to standard BGR for OpenCV processing.
        PERF: cached per simulation step — multiple callers within one step get the same array for free."""
        if not self.camera: return None
        t = self.robot.getTime()
        if getattr(self, '_bgr_cache_t', None) == t and self._bgr_cache is not None:
            return self._bgr_cache
        raw = self.camera.getImage()
        if not raw: return None
        img = cv2.cvtColor(np.frombuffer(raw, dtype=np.uint8).reshape((self.cam_h, self.cam_w, 4)), cv2.COLOR_BGRA2BGR)
        self._bgr_cache_t = t
        self._bgr_cache = img
        return img

    def get_camera_depth(self):
        """Fetches the raw float32 2D depth map from the rangefinder.
        PERF: cached per simulation step — multiple callers within one step get the same array for free."""
        if not self.rangefinder: return None
        t = self.robot.getTime()
        if getattr(self, '_depth_cache_t', None) == t and self._depth_cache is not None:
            return self._depth_cache
        raw = self.rangefinder.getRangeImage()
        if not raw: return None
        img = np.array(raw, dtype=np.float32).reshape((self.rf_h, self.rf_w))
        self._depth_cache_t = t
        self._depth_cache = img
        return img

    def get_distances_at_corners(self):
        """
        Gathers obstacle distance telemetry using Lidar and short-range IR.
        Maps the 360-degree FOV into 4 continuous, non-overlapping quadrants
        so there are ZERO blind spots at the physical corners of the robot.
        Converts front and rear quadrants to actual physical clearances from
        the robot's outer chassis boundaries.
        """
        ranges, lw, fov = self.get_lidar()
        clean = [x if not math.isinf(x) and not math.isnan(x) else 4.0 for x in ranges]

        if lw > 0:
            mid = lw // 2
            rays_per_rad = lw / fov if fov > 0 else lw / (2 * math.pi)

            
            deg_30 = int(math.radians(30) * rays_per_rad)
            deg_150 = int(math.radians(150) * rays_per_rad)

            
            f_l = min(clean[max(0, mid-deg_30) : min(lw, mid+deg_30+1)])

            
            l_l = min(clean[min(lw, mid+deg_30) : min(lw, mid+deg_150+1)]) if mid+deg_30 < lw else 4.0

            
            r_l = min(clean[max(0, mid-deg_150) : max(0, mid-deg_30+1)]) if mid-deg_30 > 0 else 4.0

            
            rr_l = 4.0
            if mid+deg_150 < lw:
                rr_l = min(rr_l, min(clean[mid+deg_150 : lw]))
            if mid-deg_150 > 0:
                rr_l = min(rr_l, min(clean[0 : mid-deg_150]))
        else:
            f_l, l_l, r_l, rr_l = 4.0, 4.0, 4.0, 4.0

        ir_f = min((s.getValue() for s in self.ir['front']), default=4.0)
        ir_r = min((s.getValue() for s in self.ir['rear']), default=4.0)

        
        
        lidar_f_clearance = max(0.0, f_l - 0.095)
        
        ir_f_clearance = max(0.0, (ir_f * 0.7071) - 0.01)
        front_clearance = min(lidar_f_clearance, ir_f_clearance)

        
        
        lidar_b_clearance = max(0.0, rr_l - 0.135)
        
        ir_b_clearance = max(0.0, (ir_r * 0.7071) - 0.01)
        rear_clearance = min(lidar_b_clearance, ir_b_clearance)

        return front_clearance, l_l, r_l, rear_clearance

    def get_exact_chassis_clearance(self):
        """
        Calculates the exact Euclidean distance from the robot's rectangular chassis
        (defined by ROBOT_HALF_L and ROBOT_HALF_W) to the nearest obstacle point
        using the full 360-degree Lidar scan and corner IR sensors.
        """
        ranges, lw, fov = self.get_lidar()

        
        points = []

        
        if lw > 0 and ranges:
            mid = lw // 2
            angle_step = fov / lw
            lidar_x = getattr(config, 'LIDAR_OFFSET_X', 0.02)
            for i, r in enumerate(ranges):
                if math.isinf(r) or math.isnan(r) or r <= 0.0:
                    continue
                theta = (i - mid) * angle_step
                px = lidar_x + r * math.cos(theta)
                py = r * math.sin(theta)
                points.append((px, py))

        
        import config as _cfg
        for key, s_list in self.ir.items():
            for idx, sensor in enumerate(s_list):
                val = sensor.getValue()
                if math.isinf(val) or math.isnan(val) or val <= 0.0 or val >= 4.0:
                    continue
                if key == 'front':
                    s_name = 'fl' if idx == 0 else 'fr'
                else:
                    s_name = 'rl' if idx == 0 else 'rr'

                if s_name in _cfg.IR_OFFSETS:
                    x_off, y_off, _, yaw_off = _cfg.IR_OFFSETS[s_name]
                    px = x_off + val * math.cos(yaw_off)
                    py = y_off + val * math.sin(yaw_off)
                    points.append((px, py))

        if not points:
            return 4.0

        
        half_l = _cfg.ROBOT_HALF_L
        half_w = _cfg.ROBOT_HALF_W

        min_dist = 4.0
        for px, py in points:
            dx = max(0.0, abs(px) - half_l)
            dy = max(0.0, abs(py) - half_w)
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def get_lidar(self):
        return self.lidar.getRangeImage(), self.lidar.getHorizontalResolution(), self.lidar.getFov()

    def get_encoders(self):
        return [s.getValue() for s in self.encoders]

    def get_imu_yaw(self):
        return self.imu.getRollPitchYaw()[2]

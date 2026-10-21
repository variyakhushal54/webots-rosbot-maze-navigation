# This file records mapping progress that should remain available during a run.
# It stores the robot path and the last confirmed positions of important pillars.
# Saved data is written in a simple format so it can be inspected outside the controller.

import os
import json
import math
import numpy as np
import cv2
from config import *

class PersistenceManager:
    def __init__(self, ):
        self.coord_history = []
        self.COORD_HIST_FILE = "coordinate_history.json"

    def _init_coord_history(self):
        """Initialize coordinates history files and states."""
        self.coord_history    = []
        self.coord_hist_confirmed = {}
        self._last_recorded_pos  = None


        if os.path.isfile(self.COORD_HIST_FILE):
            try:
                os.remove(self.COORD_HIST_FILE)
                print(f"[COORD-HIST] Reset: Old history file '{self.COORD_HIST_FILE}' removed.")
            except Exception as e:
                print(f"[COORD-HIST] WARNING: Could not remove old history file: {e}")

        print(f"[COORD-HIST] Starting fresh.")

    def _record_position(self):
        """Record current robot position in coordinates history."""

        if self._last_recorded_pos is None:
            self._last_recorded_pos = (self.loc.x, self.loc.y)
            self.coord_history.append({'x': self.loc.x, 'y': self.loc.y,
                                       'theta': self.loc.theta, 't': 0.0})
            return

        dist = math.hypot(self.loc.x - self._last_recorded_pos[0],
                          self.loc.y - self._last_recorded_pos[1])
        if dist >= COORD_RECORD_DIST:
            entry = {'x': round(self.loc.x, 4),
                     'y': round(self.loc.y, 4),
                     'theta': round(self.loc.theta, 4),
                     't': round(self.robot.getTime(), 2)}
            self.coord_history.append(entry)
            self._last_recorded_pos = (self.loc.x, self.loc.y)


            if len(self.coord_history) > COORD_HIST_MAX:
                self.coord_history = self.coord_history[-COORD_HIST_MAX:]


        for pname in ('BLUE', 'YELLOW', 'GREEN'):
            if pname in self.perception.cam_confirmed and pname not in self.coord_hist_confirmed:
                wx, wy, _ = self.perception.cam_confirmed[pname]
                self.coord_hist_confirmed[pname] = {'x': round(wx, 4), 'y': round(wy, 4)}
                print(f"[COORD-HIST] Pillar '{pname}' coordinates locked: "
                      f"({wx:.3f}, {wy:.3f})")

    def _save_coord_history(self):
        """Save history of coordinates and pillars to file."""
        total_dist = 0.0
        for i in range(1, len(self.coord_history)):
            p0, p1 = self.coord_history[i-1], self.coord_history[i]
            total_dist += math.hypot(p1['x'] - p0['x'], p1['y'] - p0['y'])

        data = {
            'waypoints':  self.coord_history,
            'pillars':    self.coord_hist_confirmed,
            'summary': {
                'total_waypoints':     len(self.coord_history),
                'total_dist_m':        round(total_dist, 3),
                'last_state':          getattr(self.state_machine, 'state', 'UNKNOWN') if hasattr(self, 'state_machine') else 'UNKNOWN',
                'last_confirmed':      list(self.perception.cam_confirmed.keys()),
                'saved_at_sim_time':   round(self.robot.getTime(), 2),
            }
        }
        tmp = self.COORD_HIST_FILE + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.COORD_HIST_FILE)
            print(f"[COORD-HIST] Saved {len(self.coord_history)} waypoints to "
                  f"'{self.COORD_HIST_FILE}' (total dist={total_dist:.1f}m)")
        except Exception as e:
            print(f"[COORD-HIST] ERROR saving history: {e}")

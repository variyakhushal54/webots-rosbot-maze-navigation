# This module keeps a compact record of places visited during the mission.
# It stores sampled robot positions and confirmed object coordinates for later use.
# File updates are spaced out and history length is capped so recording does not
# add noticeable work to the main control loop.
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
        """
        Loads the coordinate history from disk (if it exists) so the robot
        remembers previously safe positions across simulation restarts.
        Initialises self.coord_history as a list of dicts:
          {'x': float, 'y': float, 'theta': float, 't': float}
        Also initialises self.coord_hist_confirmed for confirmed pillar coords
        and self._last_recorded_pos to track distance since last record.
        """
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
        """
        Appends the current robot world-position to coord_history if the robot
        has moved at least COORD_RECORD_DIST metres since the last recorded point.
        Also merges any newly confirmed pillars into coord_hist_confirmed.
        Call this every ~30 frames from the main run() loop.
        """
        
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
        """
        Saves the recorded coordinates and confirmed pillars to disk.
        """
        data = {
            'confirmed_pillars': self.coord_hist_confirmed,
            'path': self.coord_history
        }
        try:
            with open(self.COORD_HIST_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[COORD-HIST] Saved mission history to {self.COORD_HIST_FILE}")
        except Exception as e:
            print(f"[COORD-HIST] ERROR saving history: {e}")

    def save_map(self):
        """
        Stub for map saving, kept for compatibility.
        """
        pass

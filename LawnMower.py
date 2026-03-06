import time
import threading
import numpy as np
import xml.etree.ElementTree as ET
import Telemetry_communication_agri as comm
from filelock import FileLock
from typing import Optional, List
from typing import List, Tuple, Dict, Optional
from pyproj import Geod
from dronekit import connect, VehicleMode, Command
from pymavlink import mavutil
from shapely.geometry import Polygon, LineString
_Geod = Geod(ellps="WGS84")

class LawnmowerMission:
    def __init__(self, **cfg):
        overlappedArea = 0.1        # Enter value from 0 to 1
        self.alt_of_flight = 6
        DEFAULTS = {
            "CONNECTION_STRING"  : "udp:127.0.0.1:14550",
            "BOUNDARY_FILE"      : "/home/jetsonboii/Final_agri_scan/AP_PANEL_1.kml",
            "DROPS_FILE"         : "/home/jetsonboii/Final_agri_scan/Dropping.txt",
            "WPNAV_SPEED"        : 500,        # in cm/s
            "THRESHOLD_METERS"   : 1.0,
            "SWEEP_SPACING_M"    : self.alt_of_flight*np.tan(40.5*np.pi/180)*2*(1-overlappedArea),
            "TRANSFER_DATA"      : 0.0,
            "BUFFER_M"           : 0,
            "HOVER_SECONDS"      : 1,
            "THREAD_SLEEP"       : 10,
            "METERS_PER_DEG_LAT" : 111320.0,
            "TELEMETRY_PORT"     : "/dev/ttyUSB0",
            "TELEMETRY_BAUD"     : 57600
        }
        self.cfg = DEFAULTS.copy()
        self.cfg.update(cfg)
        self.rc_rtl_triggered = False
        self.abort_reason = None
        self.telemetry_stop_event = threading.Event()
        self.telemetry_thread = None
        self.poly: Optional[Polygon] = None
        self.poly_buf: Optional[Polygon] = None
        self.path: List[Tuple[float, float]] = []
        self.dropped: List[Dict] = []
        self.arena_pts: List[Tuple[float, float]] = []
        self.vehicle= None

    def meters_to_deg_lat(self, m: float) -> float:
        return m / self.cfg["METERS_PER_DEG_LAT"]

    def haversine_m(self, lat1, lon1, lat2, lon2):
        """Calculate the great-circle distance between two points on the Earth surface."""
        az12, az21, dist = _Geod.inv(lon1, lat1, lon2, lat2)
        return dist

    def mode_callback(self, vehicle, attr_name, value):
        if value.name == "RTL":
            print("[INFO] RTL detected via RC")
            self.rc_rtl_triggered = True
            self.abort_reason = "RTL triggered by RC"
        print(f"[DEBUG] Mode changed to {value.name}")

    def parse_kml(self, kml_file):
        """Parse KML file and extract boundary coordinates"""
        """Input is KML file
            Output is lat, lon, alt"""
        try:
            tree = ET.parse(kml_file)
            root = tree.getroot()
            points = []
            for coord in root.iter():
                if coord.tag.endswith("coordinates"):
                    for c in coord.text.strip().split():
                        lon, lat, *alt = c.split(',')
                        alt = float(alt[0]) if alt else 0.0
                        points.append((float(lat), float(lon), alt))
            return points
        except Exception as e:
            print(f"Error parsing KML file: {e}")
            return []

    def make_lawnmower(self, poly: Polygon, spacing_m: float) -> List[Tuple[float, float, float]]:
        minx, miny, maxx, maxy = poly.bounds
        dlat = self.meters_to_deg_lat(spacing_m)
        path: List[Tuple[float, float, float]] = []
        toggle = False
        y = miny
        while y <= maxy + 1e-12:
            line = LineString([(minx, y), (maxx, y)])
            inter = poly.intersection(line)
            if inter.is_empty:
                y += dlat
                continue
            if inter.geom_type == "LineString":
                segs = [inter]
            elif inter.geom_type == "MultiLineString":
                segs = list(inter.geoms)
            else:
                segs = []
            for seg in segs:
                coords = list(seg.coords)
                if toggle is False:
                    ordered = coords
                else:
                    ordered = reversed(coords)
                for (lon, lat) in ordered:
                    path.append((lat, lon, float(self.alt_of_flight)))
                toggle = not toggle
            y += dlat
        return path

    def load_drops(self, filename):
        """Input is lon, lat, alt
           Output is lat, lon, alt"""
        fn = filename or self.cfg["DROPS_FILE"]
        drops = []
        # IMPORTANT: use separate lock file
        lock = FileLock(fn + ".lock")
        with lock:
            with open(fn, "r", encoding="utf-8") as f:
                for ln in f:
                    s = ln.strip()
                    if not s:
                        continue
                    parts = s.split(",")
                    if len(parts) >= 2:
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                        alt = float(parts[2].strip()) if len(parts) >= 3 else None
                        drops.append({
                            "lat": lat,
                            "lon": lon,
                            "alt": alt
                        })
        return drops

    def check_for_new_drops(self, radio_link):
        """Check for new drops in the drops file"""
        drop_list = self.load_drops(self.cfg["DROPS_FILE"])
        for drop_point in drop_list:
            if any(self.haversine_m(drop_point["lat"], drop_point["lon"], dd["lat"], dd["lon"]) < self.cfg["THRESHOLD_METERS"]
                for dd in self.dropped
            ):
                continue
            print()
            print(f"[INFO] Drop point detected at {drop_point['lat']},{drop_point['lon']}")
            # Record drop
            comm.broadcast_info(radio_link, "DROPOFF", f"{drop_point['lat']},{drop_point['lon']}")
            self.dropped.append({
                "lat": drop_point["lat"], 
                "lon": drop_point["lon"], 
                "alt": drop_point.get("alt"), 
                "time": time.time()
            })
            print(f"[INFO] Drop at new location recorded")
            print()
    
    def telemetry_worker(self, radio_link):
        """
        Runs continuously in background to broadcast telemetry data
        """
        print("[THREAD] Telemetry broadcast started")
        while not self.telemetry_stop_event.is_set():
            try:
                self.check_for_new_drops(radio_link)
                print(f"[THREAD INFO] Number of yellow spot detected are {len(self.dropped)}")
                time.sleep(self.cfg["THREAD_SLEEP"])
                if self.rc_rtl_triggered:
                    time.sleep(0.2)
                    print("[ABORT] RC RTL detected → shutting down")
                    return
            except Exception as e:
                print("[WARN][Telemetry Thread]", e)
            time.sleep(0.05)  #small sleep to avoid CPU hog
        print("[THREAD] Telemetry listener stopped")

    def connect_vehicle(self):
        print("[INFO] Connecting to vehicle...")
        try:
            self.vehicle = connect(
                self.cfg["CONNECTION_STRING"],
                wait_ready=True,
                heartbeat_timeout=120
            )
            print("[DEBUG] Connect() returned vehicle object")
        except Exception as e:
            print("[ERROR] Connect() failed:", e)
            raise
        print("[DEBUG] Waiting for heartbeat...")
        try:
            self.vehicle.wait_ready(timeout=60)
            print("[DEBUG] Vehicle ready, heartbeat OK")
        except Exception as e:
            print("[ERROR] Wait_ready() failed:", e)
            raise
        return self.vehicle

    def arm_and_takeoff(self, target_alt: float):
        if self.vehicle is None:
            raise RuntimeError("Vehicle not connected")
        print("[INFO] Arming and taking off...")
        while not self.vehicle.is_armable:
            time.sleep(0.5)
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True
        while not self.vehicle.armed:
            time.sleep(0.5)
        self.vehicle.simple_takeoff(target_alt)
        while True:
            if self.vehicle.location.global_relative_frame.alt >= target_alt * 0.9:
                break
            time.sleep(0.5)
        print("[INFO] Takeoff complete")

    def create_mission(self, waypoints, include_takeoff=True, include_rtl=True):
        """Create and upload mission to vehicle"""
        cmds = self.vehicle.commands
        cmds.clear()
        cmds.upload()
        # Add single takeoff command
        if include_takeoff:
            cmds.add(Command(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 1, 0, 0, 0, 0,
                0, 0, self.alt_of_flight
            ))
        # Add waypoints
        for wp in waypoints:
            lat = wp[0]
            lon = wp[1]
            alt = wp[2]
            cmds.add(Command(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 1,
                0, 0, 0, 0,
                lat, lon, alt
            ))
        # Add RTL (Return to Launch) command
        if include_rtl:
            cmds.add(Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 1,
                            0, 0, 0, 0, 0, 0, 0))
        cmds.upload()
        cmds.download()
        cmds.wait_ready()
        print(f"\nMission waypoints:")
        for i, c in enumerate(cmds):
            print(f"WP {i}: cmd={c.command}, lat={c.x:.6f}, lon={c.y:.6f}, alt={c.z}")
        print(f"\nMission uploaded with {len(waypoints)} waypoints")

    def finalize_mission(self) -> None:
        try:
            if self.vehicle:
                self.vehicle.parameters['RTL_ALT'] = self.alt_of_flight
                self.vehicle.mode = VehicleMode("RTL")
        except Exception:
            pass
        time.sleep(2)
        if self.vehicle:
            try:
                self.vehicle.close()
            except Exception:
                pass
        print("[INFO] Mission finished and vehicle closed.")
        import sys
        sys.exit(0)

    def run(self):
        try:
            print("="*60)
            print("Lawnmower Pattern Mission Generator")
            print("="*60)
            """Input is lon, lat, alt
               Output is lat, lon, alt which is given to create_mission"""
            print("[INFO] About to parse KML")
            radio_link = comm.start_radio(self.cfg["TELEMETRY_PORT"], self.cfg["TELEMETRY_BAUD"])
            lock = FileLock(self.cfg["DROPS_FILE"] + ".lock")
            with lock:
                open(self.cfg["DROPS_FILE"],"w").close()
                print("[INFO] Writer: File cleared")
            self.telemetry_thread = threading.Thread(
                target=self.telemetry_worker,
                args=(radio_link,),
                daemon=True
            )
            if self.cfg["BOUNDARY_FILE"].endswith('.kml'):
                self.arena_pts = [(lat, lon) for (lat, lon, *_ ) in self.parse_kml(self.cfg["BOUNDARY_FILE"])]
            else:
                print("Unsupported file format. Use .kml")
                return
            self.poly = Polygon([(lon, lat) for (lat, lon) in self.arena_pts])
            if not self.poly.is_valid:
                self.poly = self.poly.buffer(0)
                raise RuntimeError("Arena polygon invalid after buffering")
            
            self.poly_buf = self.poly.buffer(-self.meters_to_deg_lat(self.cfg["BUFFER_M"]))
            if self.poly_buf.is_empty:
                self.poly_buf = self.poly
            self.connect_vehicle()
            self.vehicle.add_attribute_listener('mode', self.mode_callback)
            path = self.make_lawnmower(self.poly_buf, self.cfg["SWEEP_SPACING_M"])
            print(f"[INFO] Generated lawnmower path with {len(path)} waypoints")

            home = self.vehicle.location.global_relative_frame
            first_wp = (home.lat, home.lon, float(self.alt_of_flight))
            path.insert(0, first_wp)
            self.create_mission(path, include_takeoff=False, include_rtl=True)

            print("Current Mode",self.vehicle.mode.name)
            print("[INFO] Waiting for pilot to ARM the vehicle...")
            while not self.vehicle.armed:
                print("  - Vehicle not armed. Awaiting RC switch...")
                time.sleep(1)
            print("[INFO] Vehicle armed by pilot!")
            self.arm_and_takeoff(self.alt_of_flight)
            print("[INFO] Switching to AUTO to run uploaded mission.")
            self.vehicle.mode = VehicleMode("AUTO")
            
            last_waypoint_index = -1
            self.vehicle.parameters['WPNAV_SPEED'] = self.cfg["WPNAV_SPEED"]
            thread_start_once = 1
            while True:
                try:
                    current_index = self.vehicle.commands.next
                    total_waypoints = self.vehicle.commands.count
                    if current_index != last_waypoint_index:
                        print(f"[INFO] Progress: Going to waypoint {current_index}/{total_waypoints}")
                        last_waypoint_index = current_index
                        if current_index/total_waypoints > self.cfg["TRANSFER_DATA"] and thread_start_once:
                            thread_start_once = 0
                            self.telemetry_thread.start()
                    if current_index >= total_waypoints:
                        print("[INFO] Mission waypoints completed, entering RTL phase.")
                        time.sleep(5)
                        break
                    if self.rc_rtl_triggered:
                        time.sleep(0.2)
                        print("[ABORT] RC RTL detected → shutting down")
                        return
                    time.sleep(1.0)
                
                except Exception as e:
                    print(f"[WARN] Error in mission loop: {e}")
                    import traceback
                    traceback.print_exc()
                    self.vehicle.mode = VehicleMode("RTL")
                    time.sleep(1.0)

        except KeyboardInterrupt:
            print("\nMission interrupted by user")
            self.vehicle.parameters['RTL_ALT'] = self.alt_of_flight
            self.vehicle.mode = VehicleMode("RTL")
        except Exception as e:
            print(f"Mission error: {e}")
            import traceback
            traceback.print_exc()
            self.vehicle.mode = VehicleMode("RTL")
        finally:
            try:
                print("[INFO] Finalizing mission data...")
                comm.broadcast_info(radio_link, "FINISH", True)
                self.finalize_mission()
            except Exception as e:
                print("[WARN] Finalize_mission raised:", e)

'''if __name__ == "__main__":
    mission = LawnmowerMission()
    mission.run()'''

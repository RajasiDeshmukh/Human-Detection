#!/usr/bin/env python3

import time
import threading
import csv
import os
from datetime import datetime
import logging

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pyproj import Transformer

try:
    from dronekit import connect
    API_HAS_DRONEKIT = True
except Exception:
    API_HAS_DRONEKIT = False


# ================= CONFIG =================

RTSP_URL = "rtsp://192.168.144.25:8554/main.264"
MODEL_PATH = "aerial_model.pt"

RESIZE_FOR_INFERENCE = (320, 192)
CONFIDENCE_THRESHOLD = 0.25

DISPLAY_WINDOW = True
FRAME_STALE_TIMEOUT = 5.0

DRONE_CONNECTION = "/dev/ttyACM0"
BAUD_RATE = 115200

HUMAN_CLASS_INDICES = {7, 8}

# ---- Camera intrinsics ----
K = np.array([
    [3933.15835, 0.0, 639.505311],
    [0.0, 3991.73499, 359.507038],
    [0.0, 0.0, 1.0]
])

# ---- Geometry ----
CAMERA_HEIGHT_M = 0.27
CLUSTER_RADIUS_M = 0.0

# ---- Logging ----
LOG_DIR = "/home/jetsonboii/flight_logs2"
VIDEO_DIR = "/home/jetsonboii/flight_logs2/videos"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

RUN_TIMESTAMP = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
CSV_PATH = os.path.join(LOG_DIR, f"human_detections_{RUN_TIMESTAMP}.csv")
VIDEO_PATH = os.path.join(VIDEO_DIR, f"detections_{RUN_TIMESTAMP}.mp4")

VIDEO_FPS = 10
VIDEO_CODEC = "mp4v"

# =========================================


latest_frame = None
last_frame_time = 0.0
grabber_running = True
frame_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


# ================= HELPERS =================

def choose_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def open_rtsp(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def frame_grabber_thread(url):
    global latest_frame, last_frame_time, grabber_running
    cap = open_rtsp(url)
    logging.info("[GRABBER] Started")

    while grabber_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            cap.release()
            cap = open_rtsp(url)
            continue

        with frame_lock:
            latest_frame = frame
            last_frame_time = time.time()

    cap.release()


def safe_vehicle_attitude(vehicle):
    try:
        att = vehicle.attitude
        return (
            np.degrees(att.roll or 0),
            np.degrees(att.pitch or 0),
            np.degrees(att.yaw or 0),
        )
    except Exception:
        return 0.0, 0.0, 0.0


def connect_drone():
    if not API_HAS_DRONEKIT:
        return None
    try:
        logging.info("[DRONE] Connecting…")
        return connect(DRONE_CONNECTION, baud=BAUD_RATE, wait_ready=False)
    except Exception:
        return None


# ================= GEOMETRY =================

def pixel2meter(u, v, H, roll, pitch, yaw):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    ray_cam = np.array([x_cam, y_cam, 1.0])

    roll, pitch, yaw = map(np.deg2rad, [roll, pitch, yaw])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])

    ray_world = (Rz @ Ry @ Rx) @ ray_cam

    if abs(ray_world[2]) < 1e-6:
        return None, None

    scale = -H / ray_world[2]
    return scale * ray_world[0], scale * ray_world[1]


def offset_to_gps(lat, lon, x_e, y_n):
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    e0, n0 = to_utm.transform(lon, lat)
    return to_wgs.transform(e0 + x_e, n0 + y_n)[::-1]


def assign_cluster(xm, ym, clusters):
    for i, c in enumerate(clusters):
        if np.hypot(xm - c["x"], ym - c["y"]) <= CLUSTER_RADIUS_M:
            return i
    return None


# ================= MAIN =================

def main():
    global grabber_running

    model = YOLO(MODEL_PATH)
    threading.Thread(target=frame_grabber_thread, args=(RTSP_URL,), daemon=True).start()

    drone = connect_drone()

    csv_file = open(CSV_PATH, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["utc_time", "lat", "lon", "confidence"])

    clusters = []
    video_writer = None

    try:
        while True:
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
                ft = last_frame_time

            if frame is None or time.time() - ft > FRAME_STALE_TIMEOUT:
                time.sleep(0.05)
                continue

            if video_writer is None:
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    VIDEO_PATH,
                    cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                    VIDEO_FPS,
                    (w, h)
                )

            small = cv2.resize(frame, RESIZE_FOR_INFERENCE)
            results = model(small, verbose=False)[0]

            sx = frame.shape[1] / RESIZE_FOR_INFERENCE[0]
            sy = frame.shape[0] / RESIZE_FOR_INFERENCE[1]

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls not in HUMAN_CLASS_INDICES or conf < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1i, y1i = int(x1 * sx), int(y1 * sy)
                x2i, y2i = int(x2 * sx), int(y2 * sy)

                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)

                if not drone:
                    continue

                px = int((x1i + x2i) / 2)
                py = y2i
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

                roll, pitch, yaw = safe_vehicle_attitude(drone)
                H = max(drone.location.global_relative_frame.alt or CAMERA_HEIGHT_M, 0.1)

                xm, ym = pixel2meter(px, py, H, roll, pitch, yaw)
                if xm is None:
                    continue

                if assign_cluster(xm, ym, clusters) is None:
                    lat, lon = offset_to_gps(
                        drone.location.global_frame.lat,
                        drone.location.global_frame.lon,
                        xm, ym
                    )

                    print("\n[HUMAN DETECTED]")
                    print(f"  Confidence : {conf:.2f}")
                    print(f"  Pixel      : ({px}, {py})")
                    print(f"  Offset (m) : x={xm:.2f}, y={ym:.2f}")
                    print(f"  GPS        : lat={lat:.6f}, lon={lon:.6f}")

                    writer.writerow([
                        datetime.utcnow().isoformat(),
                        lat, lon, round(conf, 3)
                    ])
                    csv_file.flush()

                    clusters.append({"x": xm, "y": ym})

            video_writer.write(frame)

            if DISPLAY_WINDOW:
                cv2.imshow("YOLO Drone", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

    finally:
        grabber_running = False
        csv_file.close()
        if video_writer:
            video_writer.release()
        if drone:
            drone.close()
        cv2.destroyAllWindows()
        logging.info("Shutdown complete")


if __name__ == "__main__":
    main()

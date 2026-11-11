"""Monitor whether any people are visible in one or more MediaMTX RTSP feeds.

Run: python occupancy.py
Edit config.json first. Reports OCCUPIED / EMPTY / UNKNOWN and logs transitions.
"""
import csv
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Prefer TCP transport for RTSP when using OpenCV's FFmpeg backend.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "occupancy_events.csv"


class CameraReader:
    """Read frames continuously to avoid analyzing old buffered frames."""

    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.lock = threading.Lock()
        self.frame = None
        self.last_frame_time = None
        self.last_seq = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"[{self.name}] Cannot connect; retrying in 3 seconds", flush=True)
                cap.release()
                self.stop_event.wait(3)
                continue
            print(f"[{self.name}] Connected", flush=True)
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    print(f"[{self.name}] Stream interrupted; reconnecting", flush=True)
                    break
                with self.lock:
                    self.frame = frame
                    self.last_frame_time = time.monotonic()
                    self.last_seq += 1
            cap.release()
            with self.lock:
                self.frame = None
                self.last_frame_time = None
            self.stop_event.wait(2)

    def get_latest(self):
        with self.lock:
            return self.frame, self.last_frame_time, self.last_seq

    def stop(self):
        self.stop_event.set()


def log_change(state, detail):
    new_file = not LOG_FILE.exists()
    with LOG_FILE.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp_utc", "state", "detail"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), state, detail])


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cameras = config["cameras"]
    if not cameras or any("YOUR_SERVER_IP" in url for url in cameras.values()):
        raise ValueError("Edit config.json to provide the real RTSP URLs first")

    interval = float(config.get("inference_interval_seconds", 1))
    empty_delay = float(config.get("empty_after_seconds", 15))
    stale_after = float(config.get("stale_stream_seconds", 8))
    confidence = float(config.get("confidence", 0.5))
    if interval <= 0 or empty_delay <= 0 or stale_after <= 0:
        raise ValueError("Time settings must be positive")

    print("Loading person-detection model...")
    model = YOLO(config.get("model", "yolo26n.pt"))
    readers = {name: CameraReader(name, url) for name, url in cameras.items()}
    for reader in readers.values():
        reader.start()

    # Empty is confirmed only after a continuous period of clear, fresh observations.
    clear_since = None
    last_seen = {}
    last_state = None
    last_detail = None
    last_print = 0
    display_frames = {}

    try:
        while True:
            started = time.monotonic()
            any_person = False
            all_fresh_clear = True
            details = []
            for name, reader in readers.items():
                frame, captured_at, seq = reader.get_latest()
                fresh = (frame is not None and captured_at is not None
                         and started - captured_at <= stale_after
                         and seq != last_seen.get(name))
                if not fresh:
                    all_fresh_clear = False
                    details.append(f"{name}: OFFLINE/STALE")
                    continue
                last_seen[name] = seq
                try:
                    prediction = model.predict(
                        frame, classes=[0], conf=confidence, imgsz=640,
                        verbose=False
                    )[0]
                    count = len(prediction.boxes) if prediction.boxes is not None else 0
                except Exception as exc:
                    all_fresh_clear = False
                    details.append(f"{name}: INFERENCE ERROR ({type(exc).__name__})")
                    continue
                display_frames[name] = (prediction.plot(), count)
                if count:
                    any_person = True
                details.append(f"{name}: {count} person(s)")

            now = time.monotonic()
            if any_person:
                state = "OCCUPIED"
                clear_since = None
            elif all_fresh_clear:
                if clear_since is None:
                    clear_since = now
                state = "EMPTY" if now - clear_since >= empty_delay else "UNKNOWN"
            else:
                clear_since = None
                state = "UNKNOWN"

            detail = " | ".join(details)
            if state != last_state:
                print(f"\n{datetime.now().astimezone():%Y-%m-%d %H:%M:%S} LAB: {state} ({detail})", flush=True)
                log_change(state, detail)
                last_state = state
            elif now - last_print >= 10 or detail != last_detail:
                print(f"LAB: {state} ({detail})", flush=True)
                last_print = now
            last_detail = detail

            # Display each camera's latest annotated frame without affecting CSV logging.
            for name, (annotated, count) in display_frames.items():
                canvas = annotated.copy()
                color = (0, 180, 0) if state == "OCCUPIED" else (
                    (0, 200, 255) if state == "UNKNOWN" else (220, 220, 220)
                )
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 75), (0, 0, 0), -1)
                cv2.putText(canvas, f"LAB: {state}", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(canvas, f"Visible people: {count}  |  Press Q to quit",
                            (12, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.imshow(f"Lab Camera - {name}", canvas)

            # OpenCV windows need waitKey calls to update and respond to keyboard input.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopping occupancy monitor (Q pressed)")
                break

            time.sleep(max(0, interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("\nStopping occupancy monitor")
    finally:
        for reader in readers.values():
            reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
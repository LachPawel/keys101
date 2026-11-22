#!/usr/bin/env python
"""Serve Reachy's camera as still images on this computer, to line up the view without a window.

    uv run scripts/keys_camera_serve.py              # camera 0, the one keys_cameras_midi.json uses
    uv run scripts/keys_camera_serve.py --index 1

Run it in a terminal app that may use the camera (macOS asks once). Then:
    http://127.0.0.1:8790/full.jpg    1920x1080, as the camera captures it
    http://127.0.0.1:8790/policy.jpg  640x360, exactly what the policy sees (the opencv_scaled camera)

Only this computer can reach it. Ctrl+C stops it and lets go of the camera.
"""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        raise SystemExit(f"Camera {args.index} did not open. Allow camera access for this terminal app.")
    latest: dict[str, Any] = {"frame": None}
    stop = threading.Event()

    def grab() -> None:
        while not stop.is_set():
            ok, frame = cap.read()
            if ok:
                latest["frame"] = frame
            else:
                time.sleep(0.05)

    threading.Thread(target=grab, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return

        def do_GET(self) -> None:
            frame = latest["frame"]
            if frame is None or self.path not in ("/full.jpg", "/policy.jpg"):
                self.send_response(404 if frame is not None else 503)
                self.end_headers()
                return
            if self.path == "/policy.jpg":
                frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            _ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            body = jpg.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera {args.index} at {w:.0f}x{h:.0f}: http://127.0.0.1:{args.port}/policy.jpg   (Ctrl+C stops)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        stop.set()
        cap.release()


if __name__ == "__main__":
    main()

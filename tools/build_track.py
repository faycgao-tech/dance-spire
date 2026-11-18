"""Build a Dance Spire motion track from the matching performance cut.
Usage: python tools/build_track.py VIDEO_ID source.mp4
Requires Python 3.11, mediapipe==0.10.21 and opencv-contrib-python==4.11.0.86.
Run from the repository root. Commit the generated tracks/VIDEO_ID.json.
"""
import argparse
import json
import math
import re
from pathlib import Path

import cv2
import mediapipe as mp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("video_id", type=str)
parser.add_argument("source", type=Path)
args = parser.parse_args()
if not re.fullmatch(r"[A-Za-z0-9_-]{11}", args.video_id):
    parser.error("Use the 11-character YouTube video ID.")

capture = cv2.VideoCapture(str(args.source))
fps = capture.get(cv2.CAP_PROP_FPS)
if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
    parser.error("Cannot read the source video.")
pairs = [(11,13),(13,15),(12,14),(14,16),(23,25),(25,27),(24,26),(26,28)]
samples = []
index = 0
try:
    with mp.solutions.pose.Pose(model_complexity=1) as model:
        while capture.grab():
            t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if t <= 0 and index:
                t = index / fps
            index += 1
            if t >= 30:
                break
            if t + .0001 < len(samples) / 12:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError("Could not decode video frame.")
            height, width = frame.shape[:2]
            result = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            angles = None
            if result.pose_landmarks:
                points = result.pose_landmarks.landmark
                angles = [
                    round(math.degrees(math.atan2(-(points[b].x-points[a].x)*width,
                                                   (points[b].y-points[a].y)*height)), 1)
                    if min(points[a].visibility, points[b].visibility) >= .45 else None
                    for a,b in pairs
                ]
                if sum(x is not None for x in angles) < 5:
                    angles = None
            samples.append([round(t, 3), angles])
finally:
    capture.release()
if not any(angles for _, angles in samples):
    raise SystemExit("No usable dancer poses found in the first 30 seconds.")
target = Path("tracks") / (args.video_id + ".json")
target.parent.mkdir(exist_ok=True)
tmp = target.with_suffix(".json.tmp")
tmp.write_text(json.dumps({"id":args.video_id,"samples":samples}, separators=(",",":")))
tmp.replace(target)
print(target)

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Dict, Tuple

import compiler_generated.abandoned_detection_pb2 as pb2
import compiler_generated.abandoned_detection_pb2_grpc as pb2_grpc

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

OUTPUT_PATH  = "output_ownership_grpc.avi"

PERSON_CLASS_ID = 0
FURNITURE_CLASS_IDS = {
    13: "bench", 56: "chair", 57: "couch",
    60: "dining table", 59: "bed", 61: "toilet",
}

CONFIRM_FRAMES    = 10
ABANDON_SECONDS   = 4.0
OWNERSHIP_DIST_PX = 220
STATIONARY_SECONDS        = 5.0
STATIONARY_MOVE_TOLERANCE = 25
IOU_PERSON_THRESH              = 0.10
IOU_FURN_THRESH                = 0.25
PERSON_OBJ_INTERSECTION_THRESH = 0.85
ABANDON_ZONE_RADIUS            = 80
PICKUP_TIMEOUT_SECONDS         = 3.0

C_PERSON     = (50,  205, 50)
C_OWNED      = (255, 200, 0)
C_UNATTENDED = (180, 0,   180)
C_WARNING    = (0,   140, 255)
C_ABANDONED  = (0,   0,   255)

# ─────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────

def _centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)

def _mutual_ratio(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = (max(0, min(ax2,bx2)-max(ax1,bx1))
             * max(0, min(ay2,by2)-max(ay1,by1)))
    if inter <= 0:
        return 0.0, 0.0
    aa = (ax2-ax1)*(ay2-ay1)
    ba = (bx2-bx1)*(by2-by1)
    if aa <= 0 or ba <= 0:
        return 0.0, 0.0
    return inter/aa, inter/ba

def _bbox_from_proto(bb: pb2.BoundingBox):
    return (bb.x1, bb.y1, bb.x2, bb.y2)

# ─────────────────────────────────────────────────────────────
# ABANDONED STORE
# ─────────────────────────────────────────────────────────────

@dataclass
class AbandonedRecord:
    sid: int
    locked_centroid: Tuple[int, int]
    bbox: tuple
    first_seen_ts: float
    last_seen_ts: float
    picked_up: bool = False

class AbandonedStore:

    def __init__(self):
        self._store: Dict[int, AbandonedRecord] = {}
        self._next_sid: int = 0

    @property
    def records(self):
        return self._store

    def register(self, centroid, bbox, ts) -> int:
        for sid, rec in self._store.items():
            if _dist(centroid, rec.locked_centroid) <= ABANDON_ZONE_RADIUS:
                rec.bbox         = bbox
                rec.last_seen_ts = ts
                rec.picked_up    = False
                return sid
        sid = self._next_sid
        self._next_sid += 1
        self._store[sid] = AbandonedRecord(
            sid=sid, locked_centroid=centroid, bbox=bbox,
            first_seen_ts=ts, last_seen_ts=ts,
        )
        return sid

    def find_near(self, centroid) -> Optional[int]:
        best_sid, best_d = None, float("inf")
        for sid, rec in self._store.items():
            if rec.picked_up:
                continue
            d = _dist(centroid, rec.locked_centroid)
            if d <= ABANDON_ZONE_RADIUS and d < best_d:
                best_d, best_sid = d, sid
        return best_sid

    def update_bbox(self, sid, bbox, ts):
        if sid in self._store:
            self._store[sid].bbox         = bbox
            self._store[sid].last_seen_ts = ts

    def update_visibility(self, raw_bboxes, ts):
        for rec in self._store.values():
            if rec.picked_up:
                continue
            blob_present = any(
                _dist(_centroid(bb), rec.locked_centroid) <= ABANDON_ZONE_RADIUS
                for bb in raw_bboxes
            )
            if blob_present:
                rec.last_seen_ts = ts
            else:
                gap = ts - rec.last_seen_ts
                if gap >= PICKUP_TIMEOUT_SECONDS:
                    rec.picked_up = True
                    print(f"\n[INFO] S{rec.sid} picked up after {gap:.1f}s — zone cleared.")

    def expire_picked_up(self):
        to_del = [sid for sid, rec in self._store.items() if rec.picked_up]
        for sid in to_del:
            del self._store[sid]

    def is_inside_zone(self, centroid) -> bool:
        for rec in self._store.values():
            if not rec.picked_up and _dist(centroid, rec.locked_centroid) <= ABANDON_ZONE_RADIUS:
                return True
        return False

    def __len__(self):
        return len(self._store)

# ─────────────────────────────────────────────────────────────
# BLOB TRACK
# ─────────────────────────────────────────────────────────────

@dataclass
class BlobTrack:
    tid: int
    bbox: tuple
    first_seen_ts: float
    last_seen_ts: float
    confirm_count: int = 0
    confirmed: bool = False
    ever_owned: bool = False
    owner_tid: Optional[int] = None
    owner_lost_ts: Optional[float] = None
    abandon_start: Optional[float] = None
    alert_fired: bool = False
    recent_owners: deque = field(default_factory=lambda: deque(maxlen=90))
    state: str = "confirming"
    stationary_start_ts: Optional[float] = None
    last_centroid: Optional[tuple] = None
    store_sid: int = -1

# ─────────────────────────────────────────────────────────────
# BLOB TRACKER
# ─────────────────────────────────────────────────────────────

class BlobTracker:

    def __init__(self, iou_thresh=0.25, max_missing=20):
        self.next_id     = 0
        self.iou_thresh  = iou_thresh
        self.max_missing = max_missing
        self.tracks      = {}
        self._missing    = {}

    def update(self, blobs, ts, store: AbandonedStore):
        raw_bboxes = blobs
        store.update_visibility(raw_bboxes, ts)

        det_bboxes = []
        for bb in raw_bboxes:
            sid = store.find_near(_centroid(bb))
            if sid is not None:
                store.update_bbox(sid, bb, ts)
            else:
                det_bboxes.append(bb)

        track_ids   = list(self.tracks.keys())
        assignments = {}
        matched_trk = set()

        if track_ids and det_bboxes:
            iou_mat = np.array([
                [_iou(d, self.tracks[tid].bbox) for tid in track_ids]
                for d in det_bboxes
            ])
            while iou_mat.size and iou_mat.max() >= self.iou_thresh:
                r, c = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                tid  = track_ids[c]
                if r not in assignments and tid not in matched_trk:
                    assignments[r] = tid
                    matched_trk.add(tid)
                iou_mat[r, :] = 0
                iou_mat[:, c] = 0

        active_tids = set()
        for det_idx, det_box in enumerate(det_bboxes):
            if det_idx in assignments:
                tid = assignments[det_idx]
                t   = self.tracks[tid]
                t.bbox          = det_box
                t.last_seen_ts  = ts
                t.confirm_count += 1
                if t.confirm_count >= CONFIRM_FRAMES:
                    t.confirmed = True
                self._missing[tid] = 0
            else:
                tid = self.next_id
                self.tracks[tid] = BlobTrack(
                    tid=tid, bbox=det_box,
                    first_seen_ts=ts, last_seen_ts=ts, confirm_count=1,
                )
                self._missing[tid] = 0
                self.next_id += 1
            active_tids.add(tid)

        for tid in list(self.tracks.keys()):
            if tid not in active_tids:
                self._missing[tid] = self._missing.get(tid, 0) + 1
                if self._missing[tid] > self.max_missing:
                    del self.tracks[tid]
                    self._missing.pop(tid, None)

        return [t for t in self.tracks.values() if t.confirmed]

# ─────────────────────────────────────────────────────────────
# ABANDONMENT CLASSIFIER
# ─────────────────────────────────────────────────────────────

class AbandonmentClassifier:

    def classify(self, tracks, person_bboxes, furn_bboxes, ts, frame_wh, store: AbandonedStore):
        for t in tracks:
            c = _centroid(t.bbox)

            if t.last_centroid is None:
                t.last_centroid       = c
                t.stationary_start_ts = ts
            else:
                if _dist(c, t.last_centroid) > STATIONARY_MOVE_TOLERANCE:
                    t.stationary_start_ts = ts
                t.last_centroid = c

            stat_time = ts - (t.stationary_start_ts or ts)

            if store.is_inside_zone(c):
                t.state       = "abandoned"
                t.alert_fired = True
                sid           = store.register(c, t.bbox, ts)
                t.store_sid   = sid
                continue

            if any(_iou(t.bbox, fb) >= IOU_FURN_THRESH for fb in furn_bboxes):
                t.state         = "furniture"
                t.owner_tid     = None
                t.owner_lost_ts = None
                t.abandon_start = None
                t.alert_fired   = False
                continue

            nearest_pid  = None
            nearest_dist = float("inf")

            for pid, pb in enumerate(person_bboxes):
                obj_r, per_r = _mutual_ratio(t.bbox, pb)
                if (obj_r >= PERSON_OBJ_INTERSECTION_THRESH
                        and per_r >= PERSON_OBJ_INTERSECTION_THRESH):
                    t.state     = "suppressed"
                    nearest_pid = pid
                    break
                d = _dist(c, _centroid(pb))
                if _iou(t.bbox, pb) >= IOU_PERSON_THRESH or d < OWNERSHIP_DIST_PX:
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_pid  = pid

            if t.state == "suppressed":
                continue

            if nearest_pid is not None:
                t.ever_owned    = True
                t.owner_tid     = nearest_pid
                t.owner_lost_ts = None
                t.abandon_start = None
                t.alert_fired   = False
                t.recent_owners.append((ts, nearest_pid))
                t.state = "owned"
                continue

            if not t.ever_owned:
                t.state = "unattended"
                continue

            if t.owner_lost_ts is None:
                t.owner_lost_ts = ts
            if t.abandon_start is None:
                t.abandon_start = t.owner_lost_ts

            elapsed = ts - t.abandon_start
            if elapsed >= ABANDON_SECONDS and stat_time >= STATIONARY_SECONDS:
                t.state       = "abandoned"
                t.alert_fired = True
                sid           = store.register(_centroid(t.bbox), t.bbox, ts)
                t.store_sid   = sid
            else:
                t.state = "warning"

        return tracks

# ─────────────────────────────────────────────────────────────
# PER-SESSION STATE  (one per streaming connection)
# ─────────────────────────────────────────────────────────────

class SessionState:
    """Holds all stateful components for one client session."""

    def __init__(self, W, H, fps):
        self.W   = W
        self.H   = H
        self.fps = fps
        self.tracker    = BlobTracker(iou_thresh=0.25, max_missing=25)
        self.classifier = AbandonmentClassifier()
        self.store      = AbandonedStore()
        self.baseline_furniture = []          # list of (bbox_tuple, label)
        print(f"[SERVER] New session  {W}×{H} @ {fps:.1f} fps")

    def close(self):
        print("[SERVER] Session closed")

# ─────────────────────────────────────────────────────────────
# gRPC SERVICE IMPLEMENTATION
# ─────────────────────────────────────────────────────────────

class AbandonedDetectionServicer(pb2_grpc.AbandonedDetectionServiceServicer):

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _build_response(frame_no, classified, person_bboxes, store, ts) -> pb2.FrameResponse:
        track_results = []
        for t in classified:
            bb = t.bbox
            track_results.append(pb2.TrackResult(
                tid       = t.tid,
                state     = t.state,
                bbox      = pb2.BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]),
                owner_tid = t.owner_tid if t.owner_tid is not None else -1,
                store_sid = t.store_sid,
            ))

        zones = []
        for sid, rec in store.records.items():
            bb = rec.bbox
            zones.append(pb2.AbandonedZone(
                sid           = sid,
                cx            = rec.locked_centroid[0],
                cy            = rec.locked_centroid[1],
                bbox          = pb2.BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]),
                first_seen_ts = rec.first_seen_ts,
                last_seen_ts  = rec.last_seen_ts,
                picked_up     = rec.picked_up,
                duration_seconds = ts - rec.first_seen_ts,
            ))

        return pb2.FrameResponse(
            frame_no        = frame_no,
            tracks          = track_results,
            abandoned_zones = zones,
            person_count    = len(person_bboxes),
            warning_count   = sum(1 for t in classified if t.state == "warning"),
            owned_count     = sum(1 for t in classified if t.state == "owned"),
            abandoned_count = len(store),
        )

    @staticmethod
    def _process_request(req: pb2.FrameRequest, session: SessionState):
        """Core per-frame logic shared by both RPC methods."""
        ts = req.timestamp_sec
        W, H = req.frame_width, req.frame_height

        # Parse detections from proto
        person_bboxes    = []
        live_furn_bboxes = [fb for fb, _ in session.baseline_furniture]

        for det in req.detections:
            bbox = _bbox_from_proto(det.bbox)
            if det.class_id == PERSON_CLASS_ID:
                person_bboxes.append(bbox)
            elif det.class_id in FURNITURE_CLASS_IDS:
                live_furn_bboxes.append(bbox)

        # Store baseline furniture if this is the first frame
        if req.is_first_frame:
            session.baseline_furniture = [
                (_bbox_from_proto(d.bbox), FURNITURE_CLASS_IDS.get(d.class_id, "unknown"))
                for d in req.baseline_furniture
                if d.class_id in FURNITURE_CLASS_IDS
            ]
            live_furn_bboxes = [fb for fb, _ in session.baseline_furniture]
            print(f"[SERVER] Baseline set: {len(session.baseline_furniture)} furniture items")

        # Diff blobs from client
        diff_bboxes = [_bbox_from_proto(bb) for bb in req.diff_blobs]

        # Core tracking + classification
        confirmed  = session.tracker.update(diff_bboxes, ts, session.store)
        classified = session.classifier.classify(
            confirmed, person_bboxes, live_furn_bboxes, ts, (W, H), session.store
        )
        session.store.expire_picked_up()

        return classified, person_bboxes

    # ── Bidirectional streaming RPC ───────────────────────────

    def ProcessFrameStream(self, request_iterator, context):
        session = None
        try:
            for req in request_iterator:
                # Lazy-init session on first frame
                if session is None:
                    fps_calc = (req.frame_no / req.timestamp_sec
                                if req.frame_no > 0 and req.timestamp_sec > 0 else 15.0)
                    session = SessionState(req.frame_width, req.frame_height, fps_calc)

                classified, person_bboxes = self._process_request(req, session)

                # Progress log every 30 frames
                if req.frame_no % 30 == 0:
                    sids = sorted(session.store.records.keys())
                    warn = sum(1 for t in classified if t.state == "warning")
                    print(f"[SERVER] Frame {req.frame_no:>5} | "
                          f"ts={req.timestamp_sec:.2f}s | "
                          f"persons={len(person_bboxes)} | "
                          f"abandoned={len(session.store)}(S{sids}) | "
                          f"warning={warn}")

                yield self._build_response(
                    req.frame_no, classified, person_bboxes, session.store, req.timestamp_sec
                )
        finally:
            if session:
                session.close()

    # ── Unary RPC  ────────────────────────────────────────────

    def ProcessFrame(self, request, context):
        """Stateless single-frame call — useful for testing without streaming."""
        # Create a temporary throwaway session
        session = SessionState(request.frame_width, request.frame_height, 15.0)
        classified, person_bboxes = self._process_request(request, session)
        resp = self._build_response(
            request.frame_no, classified, person_bboxes, session.store, request.timestamp_sec
        )
        session.close()
        return resp


def run_local_simulation():
    print("=== RUNNING LOCAL SIMULATION TEST FOR ABANDONED DETECTION ===")
    session = SessionState(W=1280, H=720, fps=15.0)
    
    # We will simulate 150 frames (10 seconds at 15 fps)
    fps = 15.0
    total_seconds = 10.0
    total_frames = int(total_seconds * fps)
    
    # Stationary diff blob at (200, 200, 220, 220)
    obj_bbox = (200, 200, 220, 220)
    
    print("\nSimulating tracking state transitions over 10 seconds:")
    print("-" * 90)
    print(f"{'Frame':<8} | {'Time (s)':<10} | {'Person Bbox':<28} | {'Tracked Objects State'}")
    print("-" * 90)
    
    for i in range(total_frames):
        ts = i / fps
        
        # Person bbox: Starts near the object, moves away after t=2.0s
        if ts < 2.0:
            person_bbox = (200, 200, 250, 300)
        else:
            person_bbox = (600, 600, 650, 700)
            
        person_bboxes = [person_bbox]
        diff_bboxes = [obj_bbox]
        
        # Update tracker
        confirmed = session.tracker.update(diff_bboxes, ts, session.store)
        # Classify states
        classified = session.classifier.classify(
            confirmed, person_bboxes, [], ts, (session.W, session.H), session.store
        )
        session.store.expire_picked_up()
        
        # Print info
        tracks_info = []
        for t in classified:
            tracks_info.append(f"ID {t.tid}: {t.state} (owner_tid: {t.owner_tid if t.owner_tid is not None else -1})")
            
        tracks_str = ", ".join(tracks_info) if tracks_info else "No confirmed tracks"
        
        # Print every 15 frames, or when state transitions to abandoned or warning
        is_transition = any(t.state in ["abandoned", "warning"] for t in classified)
        if i % 15 == 0 or is_transition:
            person_str = f"({person_bbox[0]}, {person_bbox[1]}) -> ({person_bbox[2]}, {person_bbox[3]})"
            print(f"{i:<8} | {ts:<10.2f} | {person_str:<28} | {tracks_str}")
            
        if len(session.store) > 0:
            print("\n[SUCCESS] Object successfully detected as ABANDONED in AbandonedStore!")
            for sid, rec in session.store.records.items():
                print(f" - Zone ID {sid}: Centroid {rec.locked_centroid}, Bbox {rec.bbox}, Duration: {ts - rec.first_seen_ts:.2f}s")
            break


def run_grpc_server():
    import grpc
    from concurrent import futures
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", 20 * 1024 * 1024),
            ("grpc.max_receive_message_length", 20 * 1024 * 1024),
        ],
    )
    pb2_grpc.add_AbandonedDetectionServiceServicer_to_server(
        AbandonedDetectionServicer(),
        server
    )
    server.add_insecure_port("0.0.0.0:50051")
    server.start()
    print("Standalone Abandoned Detection gRPC Server Running on port 50051")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop(0)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    # Ensure compiler_generated is visible on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "compiler_generated"))
    
    parser = argparse.ArgumentParser(description="Run Abandoned Inference module.")
    parser.add_argument("--server", action="store_true", help="Start the gRPC server on port 50051")
    args = parser.parse_args()
    
    if args.server:
        run_grpc_server()
    else:
        run_local_simulation()


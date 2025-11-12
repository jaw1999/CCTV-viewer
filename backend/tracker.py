"""
Object tracking for vehicle detection
Implements IoU-based tracking with Kalman filtering
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter


@dataclass
class Detection:
    """Single detection from YOLO"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class Track:
    """Tracked object"""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    age: int = 0  # Frames since last detection
    hits: int = 1  # Total number of detections
    time_since_update: int = 0


def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU score (0-1)
    """
    # Get intersection coordinates
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = bbox1_area + bbox2_area - intersection

    # Return IoU
    return intersection / union if union > 0 else 0


class VehicleTracker:
    """
    Simple multi-object tracker using IoU matching and Kalman filtering

    Based on SORT (Simple Online and Realtime Tracking) algorithm
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize tracker

        Args:
            max_age: Maximum frames to keep track alive without detections
            min_hits: Minimum detections before track is confirmed
            iou_threshold: Minimum IoU for matching detection to track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: List[Track] = []
        self.next_id = 0
        self.frame_count = 0

        # Class name mapping (COCO dataset)
        self.class_names = {
            2: 'car',
            5: 'bus',
            7: 'truck',
            3: 'motorcycle',
            1: 'bicycle',
            0: 'person'
        }

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracks with new detections

        Args:
            detections: List of Detection objects from YOLO

        Returns:
            List of confirmed tracks (hits >= min_hits)
        """
        self.frame_count += 1

        # Match detections to existing tracks using IoU
        matched_tracks, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(detections)

        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            track = self.tracks[track_idx]
            detection = detections[det_idx]

            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.class_id = detection.class_id
            track.class_name = detection.class_name
            track.hits += 1
            track.time_since_update = 0

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            new_track = Track(
                track_id=self.next_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name,
                age=0,
                hits=1,
                time_since_update=0
            )
            self.tracks.append(new_track)
            self.next_id += 1

        # Update unmatched tracks (mark as not detected this frame)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].time_since_update += 1
            self.tracks[track_idx].age += 1

        # Remove dead tracks (too old)
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        # Return only confirmed tracks
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def _match_detections_to_tracks(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU

        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t_idx, track in enumerate(self.tracks):
            for d_idx, detection in enumerate(detections):
                # Only match same class
                if track.class_id == detection.class_id:
                    iou_matrix[t_idx, d_idx] = bbox_iou(track.bbox, detection.bbox)

        # Greedy matching: highest IoU first
        matched_pairs = []
        matched_tracks = set()
        matched_detections = set()

        # Get all IoU values sorted descending
        iou_values = []
        for t_idx in range(len(self.tracks)):
            for d_idx in range(len(detections)):
                if iou_matrix[t_idx, d_idx] > self.iou_threshold:
                    iou_values.append((iou_matrix[t_idx, d_idx], t_idx, d_idx))

        iou_values.sort(reverse=True)

        # Match greedily
        for iou, t_idx, d_idx in iou_values:
            if t_idx not in matched_tracks and d_idx not in matched_detections:
                matched_pairs.append((t_idx, d_idx))
                matched_tracks.add(t_idx)
                matched_detections.add(d_idx)

        # Get unmatched
        unmatched_detections = [d_idx for d_idx in range(len(detections)) if d_idx not in matched_detections]
        unmatched_tracks = [t_idx for t_idx in range(len(self.tracks)) if t_idx not in matched_tracks]

        return matched_pairs, unmatched_detections, unmatched_tracks

    def get_track_count(self) -> Dict[str, int]:
        """Get count of confirmed tracks by vehicle type"""
        confirmed_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        counts = {}
        for track in confirmed_tracks:
            counts[track.class_name] = counts.get(track.class_name, 0) + 1
        return counts

    def reset(self):
        """Reset tracker state"""
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0


class TrackerManager:
    """Manages trackers for multiple camera feeds"""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.trackers: Dict[str, VehicleTracker] = {}
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def get_tracker(self, feed_id: str) -> VehicleTracker:
        """Get or create tracker for a feed"""
        if feed_id not in self.trackers:
            self.trackers[feed_id] = VehicleTracker(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold
            )
        return self.trackers[feed_id]

    def update_tracker(self, feed_id: str, yolo_results) -> Tuple[List[Track], Dict[str, int]]:
        """
        Update tracker for a feed with YOLO results

        Args:
            feed_id: Camera feed ID
            yolo_results: YOLO detection results object

        Returns:
            (confirmed_tracks, track_counts)
        """
        tracker = self.get_tracker(feed_id)

        # Convert YOLO results to Detection objects
        detections = []
        if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
            for box in yolo_results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()

                # Get class name
                class_name = tracker.class_names.get(class_id, f'class_{class_id}')

                detections.append(Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                ))

        # Update tracker
        confirmed_tracks = tracker.update(detections)
        track_counts = tracker.get_track_count()

        return confirmed_tracks, track_counts

    def reset_tracker(self, feed_id: str):
        """Reset tracker for a specific feed"""
        if feed_id in self.trackers:
            self.trackers[feed_id].reset()

    def reset_all(self):
        """Reset all trackers"""
        self.trackers = {}

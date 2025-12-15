"""
Unit tests for vehicle tracking module
"""
import pytest
import numpy as np
from unittest.mock import MagicMock


class TestBboxIoU:
    """Tests for bounding box IoU calculation"""

    def test_perfect_overlap(self):
        """Test IoU with identical boxes"""
        from tracker import bbox_iou

        bbox = np.array([0, 0, 100, 100])
        iou = bbox_iou(bbox, bbox)
        assert iou == pytest.approx(1.0)

    def test_no_overlap(self):
        """Test IoU with non-overlapping boxes"""
        from tracker import bbox_iou

        bbox1 = np.array([0, 0, 50, 50])
        bbox2 = np.array([100, 100, 150, 150])
        iou = bbox_iou(bbox1, bbox2)
        assert iou == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Test IoU with partially overlapping boxes"""
        from tracker import bbox_iou

        bbox1 = np.array([0, 0, 100, 100])
        bbox2 = np.array([50, 50, 150, 150])
        # Intersection: 50x50 = 2500
        # Union: 2*10000 - 2500 = 17500
        # IoU: 2500/17500 ≈ 0.1429
        iou = bbox_iou(bbox1, bbox2)
        assert iou == pytest.approx(2500 / 17500, rel=0.01)

    def test_contained_box(self):
        """Test IoU when one box contains another"""
        from tracker import bbox_iou

        outer = np.array([0, 0, 200, 200])
        inner = np.array([50, 50, 100, 100])
        # Intersection: 50x50 = 2500
        # Union: 40000 + 2500 - 2500 = 40000
        # IoU: 2500/40000 = 0.0625
        iou = bbox_iou(outer, inner)
        assert iou == pytest.approx(2500 / 40000, rel=0.01)

    def test_touching_boxes(self):
        """Test IoU with boxes that touch but don't overlap"""
        from tracker import bbox_iou

        bbox1 = np.array([0, 0, 50, 50])
        bbox2 = np.array([50, 0, 100, 50])
        iou = bbox_iou(bbox1, bbox2)
        assert iou == pytest.approx(0.0)

    def test_zero_area_box(self):
        """Test IoU with zero-area box"""
        from tracker import bbox_iou

        bbox1 = np.array([0, 0, 100, 100])
        bbox2 = np.array([50, 50, 50, 50])  # Point, no area
        iou = bbox_iou(bbox1, bbox2)
        assert iou == 0.0


class TestVehicleTracker:
    """Tests for VehicleTracker class"""

    def test_initialization(self, vehicle_tracker):
        """Test tracker initialization"""
        assert vehicle_tracker.max_age == 30
        assert vehicle_tracker.min_hits == 3
        assert vehicle_tracker.iou_threshold == 0.3
        assert len(vehicle_tracker.tracks) == 0
        assert vehicle_tracker.next_id == 0

    def test_update_creates_new_track(self, vehicle_tracker):
        """Test that update creates a new track for detection"""
        from tracker import Detection

        detection = Detection(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )

        tracks = vehicle_tracker.update([detection])

        # Track should exist but not be confirmed yet (hits < min_hits)
        assert len(vehicle_tracker.tracks) == 1
        assert vehicle_tracker.tracks[0].hits == 1
        assert len(tracks) == 0  # Not confirmed yet

    def test_track_confirmation(self, vehicle_tracker):
        """Test that track is confirmed after min_hits detections"""
        from tracker import Detection

        detection = Detection(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )

        # Update multiple times with same detection
        for i in range(3):
            tracks = vehicle_tracker.update([detection])

        # Should be confirmed now
        assert len(tracks) == 1
        assert tracks[0].hits >= 3
        assert tracks[0].class_name == "car"

    def test_track_matching_by_iou(self, vehicle_tracker):
        """Test that detections are matched to tracks by IoU"""
        from tracker import Detection

        # First detection
        det1 = Detection(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )
        vehicle_tracker.update([det1])
        initial_id = vehicle_tracker.tracks[0].track_id

        # Second detection with slight movement (high IoU)
        det2 = Detection(
            bbox=np.array([110, 110, 210, 210]),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )
        vehicle_tracker.update([det2])

        # Should still be same track
        assert len(vehicle_tracker.tracks) == 1
        assert vehicle_tracker.tracks[0].track_id == initial_id
        assert vehicle_tracker.tracks[0].hits == 2

    def test_new_track_for_different_location(self, vehicle_tracker):
        """Test that new track is created for detection at different location"""
        from tracker import Detection

        det1 = Detection(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )
        vehicle_tracker.update([det1])

        # Second detection far away (low IoU)
        det2 = Detection(
            bbox=np.array([500, 500, 600, 600]),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )
        vehicle_tracker.update([det2])

        # Should be two tracks
        assert len(vehicle_tracker.tracks) == 2

    def test_track_removal_after_max_age(self, vehicle_tracker):
        """Test that tracks are removed after max_age frames without detection"""
        from tracker import Detection

        det = Detection(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )
        vehicle_tracker.update([det])

        # Update with empty detections for max_age frames
        for _ in range(31):  # max_age + 1
            vehicle_tracker.update([])

        # Track should be removed
        assert len(vehicle_tracker.tracks) == 0

    def test_class_specific_matching(self, vehicle_tracker):
        """Test that tracks only match detections of same class"""
        from tracker import Detection

        car_det = Detection(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )
        vehicle_tracker.update([car_det])

        # Same location but different class
        truck_det = Detection(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=7,
            class_name="truck"
        )
        vehicle_tracker.update([truck_det])

        # Should be two separate tracks
        assert len(vehicle_tracker.tracks) == 2

    def test_get_track_count(self, vehicle_tracker):
        """Test getting track counts by vehicle type"""
        from tracker import Detection

        # Add multiple detections of different types
        detections = [
            Detection(np.array([100, 100, 200, 200]), 0.9, 2, "car"),
            Detection(np.array([300, 100, 400, 200]), 0.9, 2, "car"),
            Detection(np.array([500, 100, 600, 200]), 0.9, 5, "bus"),
        ]

        # Update enough times to confirm tracks
        for _ in range(3):
            vehicle_tracker.update(detections)

        counts = vehicle_tracker.get_track_count()
        assert counts.get("car", 0) == 2
        assert counts.get("bus", 0) == 1

    def test_reset(self, vehicle_tracker):
        """Test tracker reset"""
        from tracker import Detection

        det = Detection(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )
        vehicle_tracker.update([det])
        assert len(vehicle_tracker.tracks) > 0

        vehicle_tracker.reset()
        assert len(vehicle_tracker.tracks) == 0
        assert vehicle_tracker.next_id == 0
        assert vehicle_tracker.frame_count == 0


class TestTrackerManager:
    """Tests for TrackerManager class"""

    def test_get_tracker_creates_new(self, tracker_manager):
        """Test that get_tracker creates new tracker for unknown feed"""
        tracker = tracker_manager.get_tracker("feed-001")
        assert tracker is not None
        assert "feed-001" in tracker_manager.trackers

    def test_get_tracker_returns_existing(self, tracker_manager):
        """Test that get_tracker returns existing tracker"""
        tracker1 = tracker_manager.get_tracker("feed-001")
        tracker2 = tracker_manager.get_tracker("feed-001")
        assert tracker1 is tracker2

    def test_update_tracker(self, tracker_manager, sample_yolo_results):
        """Test updating tracker with YOLO results"""
        confirmed_tracks, counts = tracker_manager.update_tracker(
            "feed-001", sample_yolo_results
        )

        # First update shouldn't have confirmed tracks yet
        assert isinstance(confirmed_tracks, list)
        assert isinstance(counts, dict)

    def test_reset_tracker(self, tracker_manager):
        """Test resetting a specific tracker"""
        tracker_manager.get_tracker("feed-001")
        tracker_manager.reset_tracker("feed-001")

        # Tracker should still exist but be empty
        tracker = tracker_manager.trackers.get("feed-001")
        if tracker:
            assert len(tracker.tracks) == 0

    def test_reset_all(self, tracker_manager):
        """Test resetting all trackers"""
        tracker_manager.get_tracker("feed-001")
        tracker_manager.get_tracker("feed-002")
        assert len(tracker_manager.trackers) == 2

        tracker_manager.reset_all()
        assert len(tracker_manager.trackers) == 0


class TestDetectionDataclass:
    """Tests for Detection dataclass"""

    def test_detection_creation(self):
        """Test creating a Detection instance"""
        from tracker import Detection

        det = Detection(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )

        assert np.array_equal(det.bbox, np.array([100, 100, 200, 200]))
        assert det.confidence == 0.9
        assert det.class_id == 2
        assert det.class_name == "car"


class TestTrackDataclass:
    """Tests for Track dataclass"""

    def test_track_creation(self):
        """Test creating a Track instance"""
        from tracker import Track

        track = Track(
            track_id=1,
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )

        assert track.track_id == 1
        assert track.age == 0
        assert track.hits == 1
        assert track.time_since_update == 0

    def test_track_defaults(self):
        """Test Track default values"""
        from tracker import Track

        track = Track(
            track_id=1,
            bbox=np.array([0, 0, 0, 0]),
            confidence=0.5,
            class_id=0,
            class_name="test"
        )

        assert track.age == 0
        assert track.hits == 1
        assert track.time_since_update == 0

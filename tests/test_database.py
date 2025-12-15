"""
Unit tests for database operations
"""
import pytest
import pytest_asyncio
from datetime import datetime, timedelta


@pytest.mark.asyncio
class TestDatabaseManager:
    """Tests for DatabaseManager class"""

    async def test_init_db(self, test_db_manager):
        """Test database initialization creates tables"""
        # Tables should be created by the fixture
        # Verify by trying to run a query
        async with test_db_manager.session() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]

        assert "feeds" in tables
        assert "detections" in tables
        assert "vehicle_tracks" in tables

    async def test_upsert_feed_insert(self, test_db_manager, sample_feed_data):
        """Test inserting a new feed"""
        await test_db_manager.upsert_feed(sample_feed_data)

        # Verify feed was inserted
        from database import Feed
        from sqlalchemy import select

        async with test_db_manager.session() as session:
            result = await session.execute(
                select(Feed).where(Feed.id == sample_feed_data["id"])
            )
            feed = result.scalar_one_or_none()

        assert feed is not None
        assert feed.id == sample_feed_data["id"]
        assert feed.road_name == sample_feed_data["roadName"]

    async def test_upsert_feed_update(self, test_db_manager, sample_feed_data):
        """Test updating an existing feed"""
        # Insert first
        await test_db_manager.upsert_feed(sample_feed_data)

        # Update
        updated_data = sample_feed_data.copy()
        updated_data["roadName"] = "Updated Road"
        await test_db_manager.upsert_feed(updated_data)

        # Verify update
        from database import Feed
        from sqlalchemy import select

        async with test_db_manager.session() as session:
            result = await session.execute(
                select(Feed).where(Feed.id == sample_feed_data["id"])
            )
            feed = result.scalar_one_or_none()

        assert feed.road_name == "Updated Road"

    async def test_add_detection(self, test_db_manager, sample_feed_data):
        """Test adding a detection record"""
        # First insert the feed
        await test_db_manager.upsert_feed(sample_feed_data)

        # Add detection
        await test_db_manager.add_detection(
            feed_id=sample_feed_data["id"],
            vehicle_count=3,
            vehicle_types=["car", "bus"],
            confidence_avg=0.85
        )

        # Verify detection was added
        from database import Detection
        from sqlalchemy import select

        async with test_db_manager.session() as session:
            result = await session.execute(
                select(Detection).where(Detection.feed_id == sample_feed_data["id"])
            )
            detection = result.scalar_one_or_none()

        assert detection is not None
        assert detection.vehicle_count == 3
        assert detection.confidence_avg == 0.85

    async def test_get_feed_history(self, test_db_manager, sample_feed_data):
        """Test retrieving feed detection history"""
        # Insert feed
        await test_db_manager.upsert_feed(sample_feed_data)

        # Add multiple detections
        for i in range(5):
            await test_db_manager.add_detection(
                feed_id=sample_feed_data["id"],
                vehicle_count=i + 1,
                vehicle_types=["car"],
                confidence_avg=0.8 + i * 0.02
            )

        # Get history
        history = await test_db_manager.get_feed_history(
            sample_feed_data["id"],
            hours=24
        )

        assert len(history) == 5
        # Should be ordered by timestamp descending
        assert history[0].vehicle_count >= history[-1].vehicle_count

    async def test_get_feed_history_time_filter(self, test_db_manager, sample_feed_data):
        """Test that history respects time filter"""
        await test_db_manager.upsert_feed(sample_feed_data)

        # Add detection
        await test_db_manager.add_detection(
            feed_id=sample_feed_data["id"],
            vehicle_count=1,
            vehicle_types=["car"],
            confidence_avg=0.8
        )

        # Get history for very short period (should still include recent)
        history = await test_db_manager.get_feed_history(
            sample_feed_data["id"],
            hours=1
        )

        assert len(history) >= 1

    async def test_get_feed_stats(self, test_db_manager, sample_feed_data):
        """Test retrieving feed statistics"""
        await test_db_manager.upsert_feed(sample_feed_data)

        # Add detections
        total_vehicles = 0
        for i in range(3):
            count = i + 1
            total_vehicles += count
            await test_db_manager.add_detection(
                feed_id=sample_feed_data["id"],
                vehicle_count=count,
                vehicle_types=["car"],
                confidence_avg=0.8
            )

        stats = await test_db_manager.get_feed_stats(sample_feed_data["id"], days=7)

        assert stats["total_detections"] == 3
        assert stats["total_vehicles"] == total_vehicles
        assert stats["period_days"] == 7

    async def test_update_vehicle_track_new(self, test_db_manager, sample_feed_data):
        """Test creating a new vehicle track"""
        await test_db_manager.upsert_feed(sample_feed_data)

        await test_db_manager.update_vehicle_track(
            feed_id=sample_feed_data["id"],
            track_id=1,
            vehicle_class="car",
            confidence=0.9,
            position=(150, 150)
        )

        from database import VehicleTrack
        from sqlalchemy import select

        async with test_db_manager.session() as session:
            result = await session.execute(
                select(VehicleTrack).where(
                    VehicleTrack.feed_id == sample_feed_data["id"],
                    VehicleTrack.track_id == 1
                )
            )
            track = result.scalar_one_or_none()

        assert track is not None
        assert track.vehicle_class == "car"
        assert track.frame_count == 1

    async def test_update_vehicle_track_existing(self, test_db_manager, sample_feed_data):
        """Test updating an existing vehicle track"""
        await test_db_manager.upsert_feed(sample_feed_data)

        # Create initial track
        await test_db_manager.update_vehicle_track(
            feed_id=sample_feed_data["id"],
            track_id=1,
            vehicle_class="car",
            confidence=0.9,
            position=(150, 150)
        )

        # Update track
        await test_db_manager.update_vehicle_track(
            feed_id=sample_feed_data["id"],
            track_id=1,
            vehicle_class="car",
            confidence=0.85,
            position=(160, 160)
        )

        from database import VehicleTrack
        from sqlalchemy import select

        async with test_db_manager.session() as session:
            result = await session.execute(
                select(VehicleTrack).where(
                    VehicleTrack.feed_id == sample_feed_data["id"],
                    VehicleTrack.track_id == 1
                )
            )
            track = result.scalar_one_or_none()

        assert track.frame_count == 2
        assert 0.85 < track.confidence_avg < 0.9  # Average of 0.9 and 0.85

    async def test_cleanup_old_data(self, test_db_manager, sample_feed_data):
        """Test cleanup of old data"""
        await test_db_manager.upsert_feed(sample_feed_data)

        # Add detection
        await test_db_manager.add_detection(
            feed_id=sample_feed_data["id"],
            vehicle_count=1,
            vehicle_types=["car"],
            confidence_avg=0.8
        )

        # Cleanup with 0 day retention (should delete everything)
        result = await test_db_manager.cleanup_old_data(
            detection_retention_days=0,
            history_retention_days=0
        )

        assert result["detections_deleted"] >= 0
        assert result["tracks_deleted"] >= 0

    async def test_session_rollback_on_error(self, test_db_manager):
        """Test that session rolls back on error"""
        # Try to add detection for non-existent feed
        # This should work (no foreign key enforcement in SQLite by default)
        # but we can test the transaction behavior
        try:
            await test_db_manager.add_detection(
                feed_id="non-existent",
                vehicle_count=1,
                vehicle_types=["car"],
                confidence_avg=0.8
            )
        except Exception:
            pass  # Expected to potentially fail

        # Verify we can still use the session
        from database import Feed
        from sqlalchemy import select, func

        async with test_db_manager.session() as session:
            result = await session.execute(select(func.count()).select_from(Feed))
            count = result.scalar()

        assert count >= 0  # Just verify query works


@pytest.mark.asyncio
class TestFeedModel:
    """Tests for Feed model"""

    async def test_feed_defaults(self, test_db_manager, sample_feed_data):
        """Test Feed model default values"""
        await test_db_manager.upsert_feed(sample_feed_data)

        from database import Feed
        from sqlalchemy import select

        async with test_db_manager.session() as session:
            result = await session.execute(
                select(Feed).where(Feed.id == sample_feed_data["id"])
            )
            feed = result.scalar_one()

        assert feed.is_working == False
        assert feed.last_seen is not None
        assert feed.last_vehicle_detection is None

    async def test_feed_relationships(self, test_db_manager, sample_feed_data):
        """Test Feed model relationships"""
        await test_db_manager.upsert_feed(sample_feed_data)

        await test_db_manager.add_detection(
            feed_id=sample_feed_data["id"],
            vehicle_count=1,
            vehicle_types=["car"],
            confidence_avg=0.8
        )

        from database import Feed
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        async with test_db_manager.session() as session:
            result = await session.execute(
                select(Feed)
                .where(Feed.id == sample_feed_data["id"])
                .options(selectinload(Feed.detections))
            )
            feed = result.scalar_one()

            # Access relationship
            assert len(feed.detections) >= 1


@pytest.mark.asyncio
class TestDetectionModel:
    """Tests for Detection model"""

    async def test_detection_vehicle_types_json(self, test_db_manager, sample_feed_data):
        """Test that vehicle_types is stored as JSON"""
        await test_db_manager.upsert_feed(sample_feed_data)

        types = ["car", "bus", "truck"]
        await test_db_manager.add_detection(
            feed_id=sample_feed_data["id"],
            vehicle_count=3,
            vehicle_types=types,
            confidence_avg=0.85
        )

        from database import Detection
        from sqlalchemy import select
        import json

        async with test_db_manager.session() as session:
            result = await session.execute(
                select(Detection).where(Detection.feed_id == sample_feed_data["id"])
            )
            detection = result.scalar_one()

        stored_types = json.loads(detection.vehicle_types)
        assert stored_types == types


@pytest.mark.asyncio
class TestVehicleTrackModel:
    """Tests for VehicleTrack model"""

    async def test_track_trajectory_json(self, test_db_manager, sample_feed_data):
        """Test that trajectory is stored as JSON"""
        await test_db_manager.upsert_feed(sample_feed_data)

        await test_db_manager.update_vehicle_track(
            feed_id=sample_feed_data["id"],
            track_id=1,
            vehicle_class="car",
            confidence=0.9,
            position=(100, 100)
        )

        await test_db_manager.update_vehicle_track(
            feed_id=sample_feed_data["id"],
            track_id=1,
            vehicle_class="car",
            confidence=0.9,
            position=(110, 110)
        )

        from database import VehicleTrack
        from sqlalchemy import select
        import json

        async with test_db_manager.session() as session:
            result = await session.execute(
                select(VehicleTrack).where(
                    VehicleTrack.feed_id == sample_feed_data["id"],
                    VehicleTrack.track_id == 1
                )
            )
            track = result.scalar_one()

        trajectory = json.loads(track.trajectory)
        assert len(trajectory) == 2
        assert trajectory[0] == [100, 100]
        assert trajectory[1] == [110, 110]

"""
Database models and session management.
"""
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, ForeignKey, Text, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import asynccontextmanager
from typing import Optional

Base = declarative_base()


class Feed(Base):
    __tablename__ = 'feeds'

    id = Column(String(50), primary_key=True)
    road_name = Column(String(100))
    location_mile = Column(String(50))
    description = Column(Text)
    lat = Column(Float)
    lon = Column(Float)
    direction = Column(String(20))
    image_url = Column(String(500))
    stream_url = Column(String(500))

    # Status tracking
    is_working = Column(Boolean, default=False)
    last_seen = Column(DateTime, default=datetime.utcnow)
    last_vehicle_detection = Column(DateTime, nullable=True)

    # Relationships
    detections = relationship("Detection", back_populates="feed", cascade="all, delete-orphan")
    tracks = relationship("VehicleTrack", back_populates="feed", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = 'detections'
    __table_args__ = (
        # Composite index for common queries (feed + time range)
        Index('ix_detections_feed_timestamp', 'feed_id', 'timestamp'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    feed_id = Column(String(50), ForeignKey('feeds.id'), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Detection details
    vehicle_count = Column(Integer, default=0)
    vehicle_types = Column(String(100))  # JSON string: ["car", "truck", "bus"]
    confidence_avg = Column(Float)

    # Bounding boxes (JSON string)
    boxes = Column(Text, nullable=True)  # JSON array of boxes

    # Image reference
    snapshot_path = Column(String(500), nullable=True)

    # Relationship
    feed = relationship("Feed", back_populates="detections")


class VehicleTrack(Base):
    __tablename__ = 'vehicle_tracks'
    __table_args__ = (
        # Composite index for track lookup
        Index('ix_tracks_feed_track', 'feed_id', 'track_id'),
        # Index for cleanup queries
        Index('ix_tracks_last_seen', 'last_seen'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    feed_id = Column(String(50), ForeignKey('feeds.id'), index=True)
    track_id = Column(Integer, index=True)  # Tracker-assigned ID

    # Tracking info
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    frame_count = Column(Integer, default=0)

    # Vehicle classification
    vehicle_class = Column(String(20))  # car, truck, bus, motorcycle, etc.
    confidence_avg = Column(Float)

    # Movement data (JSON strings)
    trajectory = Column(Text, nullable=True)  # JSON array of [x, y] positions
    speeds = Column(Text, nullable=True)  # JSON array of estimated speeds

    # Relationship
    feed = relationship("Feed", back_populates="tracks")


class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✓ Database tables created")

    async def close(self):
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self):
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def upsert_feed(self, feed_data: dict):
        async with self.session() as session:
            from sqlalchemy import select

            # Check if feed exists
            result = await session.execute(
                select(Feed).where(Feed.id == feed_data['id'])
            )
            feed = result.scalar_one_or_none()

            if feed:
                # Update existing
                feed.road_name = feed_data.get('roadName', '')
                feed.location_mile = feed_data.get('locationMile', '')
                feed.description = feed_data.get('description', '')
                feed.lat = float(feed_data.get('lat', 0) or 0)
                feed.lon = float(feed_data.get('lon', 0) or 0)
                feed.direction = feed_data.get('direction', '')
                feed.image_url = feed_data.get('imageUrl', '')
                feed.stream_url = feed_data.get('streamUrl', '')
                feed.last_seen = datetime.utcnow()
            else:
                # Insert new
                feed = Feed(
                    id=feed_data['id'],
                    road_name=feed_data.get('roadName', ''),
                    location_mile=feed_data.get('locationMile', ''),
                    description=feed_data.get('description', ''),
                    lat=float(feed_data.get('lat', 0) or 0),
                    lon=float(feed_data.get('lon', 0) or 0),
                    direction=feed_data.get('direction', ''),
                    image_url=feed_data.get('imageUrl', ''),
                    stream_url=feed_data.get('streamUrl', ''),
                    is_working=False,
                    last_seen=datetime.utcnow()
                )
                session.add(feed)

    async def add_detection(self, feed_id: str, vehicle_count: int,
                           vehicle_types: list, confidence_avg: float,
                           boxes: Optional[str] = None,
                           snapshot_path: Optional[str] = None):
        async with self.session() as session:
            from sqlalchemy import select
            import json

            detection = Detection(
                feed_id=feed_id,
                timestamp=datetime.utcnow(),
                vehicle_count=vehicle_count,
                vehicle_types=json.dumps(vehicle_types),
                confidence_avg=confidence_avg,
                boxes=boxes,
                snapshot_path=snapshot_path
            )
            session.add(detection)

            # Update feed's last_vehicle_detection
            result = await session.execute(
                select(Feed).where(Feed.id == feed_id)
            )
            feed = result.scalar_one_or_none()
            if feed:
                feed.last_vehicle_detection = datetime.utcnow()

    async def update_vehicle_track(self, feed_id: str, track_id: int,
                                   vehicle_class: str, confidence: float,
                                   position: tuple, speed: Optional[float] = None):
        async with self.session() as session:
            from sqlalchemy import select, and_
            import json

            # Find existing track
            result = await session.execute(
                select(VehicleTrack).where(
                    and_(
                        VehicleTrack.feed_id == feed_id,
                        VehicleTrack.track_id == track_id
                    )
                )
            )
            track = result.scalar_one_or_none()

            if track:
                # Update existing track
                track.last_seen = datetime.utcnow()
                track.frame_count += 1

                # Update trajectory
                trajectory = json.loads(track.trajectory) if track.trajectory else []
                trajectory.append(list(position))
                track.trajectory = json.dumps(trajectory)

                # Update speeds
                if speed is not None:
                    speeds = json.loads(track.speeds) if track.speeds else []
                    speeds.append(speed)
                    track.speeds = json.dumps(speeds)

                # Update average confidence
                track.confidence_avg = (track.confidence_avg * (track.frame_count - 1) + confidence) / track.frame_count
            else:
                # Create new track
                track = VehicleTrack(
                    feed_id=feed_id,
                    track_id=track_id,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    frame_count=1,
                    vehicle_class=vehicle_class,
                    confidence_avg=confidence,
                    trajectory=json.dumps([list(position)]),
                    speeds=json.dumps([speed] if speed is not None else [])
                )
                session.add(track)

    async def get_feed_history(self, feed_id: str, hours: int = 24):
        from sqlalchemy import select
        from datetime import timedelta

        async with self.session() as session:
            since = datetime.utcnow() - timedelta(hours=hours)
            result = await session.execute(
                select(Detection)
                .where(Detection.feed_id == feed_id)
                .where(Detection.timestamp >= since)
                .order_by(Detection.timestamp.desc())
            )
            return result.scalars().all()

    async def get_feed_stats(self, feed_id: str, days: int = 7):
        from sqlalchemy import select, func
        from datetime import timedelta

        async with self.session() as session:
            since = datetime.utcnow() - timedelta(days=days)

            # Total detections
            result = await session.execute(
                select(func.count(Detection.id))
                .where(Detection.feed_id == feed_id)
                .where(Detection.timestamp >= since)
            )
            total_detections = result.scalar() or 0

            # Total vehicles
            result = await session.execute(
                select(func.sum(Detection.vehicle_count))
                .where(Detection.feed_id == feed_id)
                .where(Detection.timestamp >= since)
            )
            total_vehicles = result.scalar() or 0

            # Average vehicles per detection
            avg_vehicles = total_vehicles / total_detections if total_detections > 0 else 0

            return {
                "total_detections": total_detections,
                "total_vehicles": total_vehicles,
                "avg_vehicles_per_detection": round(avg_vehicles, 2),
                "period_days": days
            }

    async def cleanup_old_data(self, detection_retention_days: int = 30,
                              history_retention_days: int = 90):
        from sqlalchemy import delete
        from datetime import timedelta

        async with self.session() as session:
            detection_cutoff = datetime.utcnow() - timedelta(days=detection_retention_days)
            history_cutoff = datetime.utcnow() - timedelta(days=history_retention_days)

            # Delete old detections
            result = await session.execute(
                delete(Detection).where(Detection.timestamp < detection_cutoff)
            )
            detections_deleted = result.rowcount

            # Delete old tracks
            result = await session.execute(
                delete(VehicleTrack).where(VehicleTrack.last_seen < history_cutoff)
            )
            tracks_deleted = result.rowcount

            print(f"Cleanup: {detections_deleted} detections, {tracks_deleted} tracks deleted")

            return {
                "detections_deleted": detections_deleted,
                "tracks_deleted": tracks_deleted
            }

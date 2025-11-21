#!/usr/bin/env python3
"""
FastAPI backend for Taiwan CCTV Viewer
Ingests and caches all 2,500+ feeds in real-time
"""

import asyncio
import ssl
import sys
import time
import socket
import uuid
import hashlib
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import numpy as np

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from PIL import Image
from ultralytics import YOLO
from pydantic import BaseModel
import cv2

# Import our new modules
try:
    from .database import DatabaseManager
    from .tracker import TrackerManager, Detection as TrackerDetection
    from .websocket_manager import ConnectionManager, handle_websocket_messages
    from .observability import (
        MetricsCollector, HealthChecker, StructuredLogger,
        CircuitBreaker, AlertManager
    )
except ImportError:
    # Fallback for direct execution
    from database import DatabaseManager
    from tracker import TrackerManager, Detection as TrackerDetection
    from websocket_manager import ConnectionManager, handle_websocket_messages
    from observability import (
        MetricsCollector, HealthChecker, StructuredLogger,
        CircuitBreaker, AlertManager
    )

# Import Prometheus client for metrics endpoint
from prometheus_client import make_asgi_app

# Load configuration
def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if not config_path.exists():
        print(f"Warning: config.yaml not found at {config_path}, using defaults")
        return {}

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"✓ Configuration loaded from {config_path}")
    return config

# Load config at module level
CONFIG = load_config()

# Global state
feeds_data: List[Dict] = []
feed_cache: Dict[str, bytes] = {}  # feed_id -> latest JPEG bytes
feed_status: Dict[str, bool] = {}  # feed_id -> is_working
feed_vehicle_detected: Dict[str, bool] = {}  # feed_id -> has_vehicles
feed_image_hash: Dict[str, str] = {}  # feed_id -> image hash for change detection
feed_retry_after: Dict[str, float] = {}  # feed_id -> next_try_time for exponential backoff
last_update: float = 0
cycle_counter: int = 0  # For selective detection cycles

# YOLO model (loaded at startup)
yolo_model = None
# ThreadPoolExecutor for offloading CPU-intensive YOLO inference
executor = None

# Database manager
db_manager: Optional[DatabaseManager] = None

# Tracker manager
tracker_manager: Optional[TrackerManager] = None

# WebSocket manager
ws_manager = ConnectionManager()

# Observability components
metrics = MetricsCollector()
logger = StructuredLogger("cctv.main", log_level=CONFIG.get('logging', {}).get('level', 'INFO'))
alert_manager = AlertManager(CONFIG)
feed_circuit_breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=300.0)
health_checker = None  # Initialized in lifespan

# Detection settings from config
VEHICLE_CLASSES = CONFIG.get('detection', {}).get('vehicle_classes', [2, 5, 7])
MIN_BOX_SIZE = CONFIG.get('detection', {}).get('min_box_size', 20)
CONFIDENCE_THRESHOLD = CONFIG.get('detection', {}).get('confidence_threshold', 0.6)
SELECTIVE_SKIP_INTERVAL = CONFIG.get('performance', {}).get('selective_skip_interval', 2)

# Stream Out configuration from config
stream_config = {
    "enabled": CONFIG.get('stream_out', {}).get('enabled', False),
    "format": CONFIG.get('stream_out', {}).get('format', 'cot'),
    "ip": CONFIG.get('stream_out', {}).get('cot', {}).get('ip', '127.0.0.1'),
    "port": CONFIG.get('stream_out', {}).get('cot', {}).get('port', 8087),
    "latticeToken": CONFIG.get('stream_out', {}).get('lattice', {}).get('token', ''),
    "latticeSandboxToken": CONFIG.get('stream_out', {}).get('lattice', {}).get('sandbox_token', ''),
    "latticeIntegration": CONFIG.get('stream_out', {}).get('lattice', {}).get('integration_name', 'taiwan-cctv'),
    "latticeUrl": CONFIG.get('stream_out', {}).get('lattice', {}).get('url', '')
}

# Pydantic model for stream configuration
class StreamConfig(BaseModel):
    enabled: bool
    format: str
    ip: str
    port: int
    latticeToken: Optional[str] = ""
    latticeSandboxToken: Optional[str] = ""
    latticeIntegration: Optional[str] = "taiwan-cctv"
    latticeUrl: Optional[str] = ""

# Lattice client (initialized when token is provided)
lattice_client = None

# SSL context for Taiwan servers
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


def generate_cot_message(feed: Dict) -> str:
    """Generate a CoT XML message for a camera with vehicle detection"""
    # Generate ISO 8601 timestamp with milliseconds
    now = datetime.now(timezone.utc)
    time_str = now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    # Stale time (1 hour from now)
    stale_time = datetime.fromtimestamp(now.timestamp() + 3600, timezone.utc)
    stale_str = stale_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    # Use feed ID as UID
    uid = feed['id']

    # Callsign: TrafficCam-{ID}
    callsign = f"TrafficCam-{feed['id']}"

    # Build remarks with location info
    remarks_parts = []
    if feed.get('roadName'):
        remarks_parts.append(f"Road: {feed['roadName']}")
    if feed.get('locationMile'):
        remarks_parts.append(f"Location: {feed['locationMile']}")
    if feed.get('description'):
        remarks_parts.append(feed['description'])
    remarks_parts.append("Vehicle Detected")
    remarks = " | ".join(remarks_parts)

    # Get lat/lon
    lat = feed.get('lat', '0.0')
    lon = feed.get('lon', '0.0')

    # Build CoT XML
    cot_xml = f'''<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<event version='2.0' uid='{uid}' type='a-u-G' time='{time_str}' start='{time_str}' stale='{stale_str}' how='h-e'>
    <point lat='{lat}' lon='{lon}' hae='9999999.0' ce='9999999.0' le='9999999.0' />
    <detail>
        <contact callsign='{callsign}'/>
        <remarks>{remarks}</remarks>
        <link uid='{uid}' type='video' url='http://localhost:8001/api/feeds/{uid}/snapshot'/>
        <usericon iconsetpath='COT_MAPPING_2525B/a-u/a-u-G'/>
    </detail>
</event>'''

    return cot_xml


def send_cot_udp(message: str, ip: str, port: int):
    """Send CoT message via UDP"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(message.encode('utf-8'), (ip, port))
        sock.close()
    except Exception as e:
        print(f"Error sending CoT message: {e}")


def publish_lattice_entity(feed: Dict, integration_name: str, annotated_image: bytes = None):
    """Publish entity to Lattice with vehicle detection and image"""
    global lattice_client

    if lattice_client is None:
        print("⚠ Lattice client is None, skipping publish")
        return

    try:
        from anduril import Location, Position, Aliases, Provenance, Ontology, MilView, Media, MediaItem, Enu
        from datetime import timedelta
        import math

        # Parse lat/lon
        lat = float(feed.get('lat', '0.0'))
        lon = float(feed.get('lon', '0.0'))

        # Create entity name/description
        name_parts = []
        if feed.get('roadName'):
            name_parts.append(feed['roadName'])
        if feed.get('locationMile'):
            name_parts.append(feed['locationMile'])
        entity_name = " - ".join(name_parts) if name_parts else feed['id']

        description = f"Traffic Camera: {entity_name} | Vehicle Detected"

        now = datetime.now(timezone.utc)
        expiry = now + timedelta(hours=1)

        entity_id = feed['id']
        entity_name_alias = f"TrafficCam-{entity_id}"

        print(f"Publishing entity {entity_id} to Lattice...")
        print(f"  Location: ({lat}, {lon})")
        print(f"  Description: {description}")
        print(f"  Alias: {entity_name_alias}")

        # Upload detection image to Objects API if provided
        media = None
        if annotated_image:
            try:
                # IMPORTANT: Use flat naming (no slashes) - slashes don't work in sandbox
                # Format: integration-entity_id-timestamp.jpg
                object_path = f"{integration_name}-{entity_id}-{now.strftime('%Y%m%d_%H%M%S')}.jpg"

                lattice_client.objects.upload_object(
                    object_path=object_path,
                    request=annotated_image
                )

                # Reference the uploaded object in Media
                # MediaItemType is a string literal union, use "MEDIA_TYPE_IMAGE" directly
                # relative_path should be the path relative to environment base URL
                media = Media(
                    media=[
                        MediaItem(
                            type="MEDIA_TYPE_IMAGE",
                            relative_path=f"api/v1/objects/{object_path}"
                        )
                    ]
                )
                print(f"  📷 Uploaded detection image to {object_path} ({len(annotated_image)} bytes)")
            except Exception as img_error:
                print(f"  ⚠ Failed to upload image: {img_error}")

        # Calculate unit velocity vector based on road direction for heading indication
        # We only know direction, not speed, so use unit vector (magnitude=1) for heading only
        direction = feed.get('direction', '').upper()
        velocity_enu = None
        if direction:
            # Map direction to unit velocity vector (magnitude = 1.0)
            # ENU: East-North-Up coordinate system
            # Using unit vector since we don't have actual speed data
            direction_map = {
                'E': Enu(e=1.0, n=0.0, u=0.0),      # East
                'W': Enu(e=-1.0, n=0.0, u=0.0),     # West
                'N': Enu(e=0.0, n=1.0, u=0.0),      # North
                'S': Enu(e=0.0, n=-1.0, u=0.0),     # South
                'NE': Enu(e=1.0/math.sqrt(2), n=1.0/math.sqrt(2), u=0.0),
                'NW': Enu(e=-1.0/math.sqrt(2), n=1.0/math.sqrt(2), u=0.0),
                'SE': Enu(e=1.0/math.sqrt(2), n=-1.0/math.sqrt(2), u=0.0),
                'SW': Enu(e=-1.0/math.sqrt(2), n=-1.0/math.sqrt(2), u=0.0),
            }
            velocity_enu = direction_map.get(direction)
            if velocity_enu:
                print(f"  🧭 Added heading: {direction}")

        # Publish entity using keyword arguments
        response = lattice_client.entities.publish_entity(
            entity_id=entity_id,
            description=description,
            is_live=True,
            created_time=now,
            expiry_time=expiry,
            location=Location(
                position=Position(
                    latitude_degrees=lat,
                    longitude_degrees=lon,
                    altitude_hae_meters=0.0
                ),
                velocity_enu=velocity_enu  # Unit vector for heading only, no speed data
            ),
            aliases=Aliases(
                name=entity_name_alias
            ),
            ontology=Ontology(
                template="TEMPLATE_TRACK",
                platform_type="VEHICLE"
            ),
            mil_view=MilView(
                disposition="DISPOSITION_UNKNOWN",
                environment="ENVIRONMENT_SURFACE"
            ),
            provenance=Provenance(
                integration_name=integration_name,
                data_type="CCTV",
                source_update_time=now
            ),
            media=media
        )

        print(f"✓ Successfully published entity {entity_id}")
    except Exception as e:
        print(f"✗ Error publishing to Lattice: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")


def initialize_lattice_client(env_token: str, sandbox_token: str = None, base_url: str = None):
    """
    Initialize Lattice client with tokens

    For Sandboxes:
    - env_token: Environment-level "Lattice Auth Token" (goes in Authorization: Bearer header)
    - sandbox_token: Account-level Sandboxes token (goes in anduril-sandbox-authorization header)

    For Production:
    - env_token: Your production Lattice token
    - sandbox_token: Not needed
    """
    global lattice_client
    try:
        from anduril import Lattice

        # Debug: Check if tokens are provided
        env_token_preview = f"{env_token[:10]}..." if len(env_token) > 10 else "EMPTY/SHORT"
        print(f"Initializing Lattice with environment token: {env_token_preview}")

        # Ensure base_url uses HTTPS
        if base_url and base_url.startswith("http://"):
            base_url = base_url.replace("http://", "https://")
            print(f"⚠ Fixed base_url protocol to HTTPS: {base_url}")

        # For sandbox environments, we need BOTH tokens
        # Use the SDK's headers parameter to pass the sandbox token
        custom_headers = {}
        if base_url and "sandbox" in base_url.lower():
            if sandbox_token:
                sandbox_token_preview = f"{sandbox_token[:10]}..." if len(sandbox_token) > 10 else "EMPTY/SHORT"
                print(f"Detected sandbox environment - using sandbox token: {sandbox_token_preview}")
                # IMPORTANT: Sandbox token also needs "Bearer " prefix!
                custom_headers["anduril-sandbox-authorization"] = f"Bearer {sandbox_token}"
            else:
                print("⚠ Warning: Sandbox URL detected but no sandbox token provided!")

        # Initialize with SDK's native headers parameter (better than custom httpx client)
        if custom_headers:
            if base_url:
                print(f"Using base_url: {base_url}")
                lattice_client = Lattice(
                    token=env_token,
                    base_url=base_url,
                    headers=custom_headers
                )
            else:
                lattice_client = Lattice(
                    token=env_token,
                    headers=custom_headers
                )
        else:
            # Standard initialization for production environments
            if base_url:
                print(f"Using base_url: {base_url}")
                lattice_client = Lattice(token=env_token, base_url=base_url)
            else:
                print("Using default Lattice URL")
                lattice_client = Lattice(token=env_token)

        print(f"✓ Lattice client initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing Lattice client: {e}")
        lattice_client = None


async def fetch_feed_list():
    """Fetch and parse the XML feed list from Taiwan Highway Bureau"""
    global feeds_data, last_update

    url = 'https://cctv-maintain.thb.gov.tw/opendataCCTVs.xml'

    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        try:
            response = await client.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()

            # Parse XML
            root = ET.fromstring(response.text)

            # Extract namespace
            ns = {'ns': 'http://traffic.transportdata.tw/standard/traffic/schema/'}

            feeds = []
            for cctv in root.findall('.//ns:CCTV', ns):
                feed = {
                    'id': cctv.find('ns:CCTVID', ns).text if cctv.find('ns:CCTVID', ns) is not None else '',
                    'streamUrl': cctv.find('ns:VideoStreamURL', ns).text if cctv.find('ns:VideoStreamURL', ns) is not None else '',
                    'imageUrl': cctv.find('ns:VideoImageURL', ns).text if cctv.find('ns:VideoImageURL', ns) is not None else '',
                    'description': cctv.find('ns:SurveillanceDescription', ns).text if cctv.find('ns:SurveillanceDescription', ns) is not None else '',
                    'roadName': cctv.find('ns:RoadName', ns).text if cctv.find('ns:RoadName', ns) is not None else '',
                    'locationMile': cctv.find('ns:LocationMile', ns).text if cctv.find('ns:LocationMile', ns) is not None else '',
                    'lat': cctv.find('ns:PositionLat', ns).text if cctv.find('ns:PositionLat', ns) is not None else '',
                    'lon': cctv.find('ns:PositionLon', ns).text if cctv.find('ns:PositionLon', ns) is not None else '',
                    'direction': cctv.find('ns:RoadDirection', ns).text if cctv.find('ns:RoadDirection', ns) is not None else '',
                }
                if feed['id'] and feed['imageUrl']:
                    feeds.append(feed)

            feeds_data = feeds
            last_update = time.time()

            print(f"Loaded {len(feeds)} feeds from Taiwan Highway Bureau")
            return feeds

        except Exception as e:
            print(f"Error fetching feed list: {e}")
            return []


def detect_vehicles(img_bytes: bytes, feed_id: str = None) -> tuple[bool, bytes, dict]:
    """Detect if vehicles are present in image using YOLO with tracking (optimized with OpenCV)
    Returns: (has_vehicles, annotated_image_bytes, detection_data)
    """
    try:
        if yolo_model is None:
            return False, img_bytes, {}

        # OPTIMIZATION: Use OpenCV for faster decoding (2-3x faster than PIL)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_array is None:
            return False, img_bytes, {}

        # Run YOLO inference with confidence threshold
        results = yolo_model(img_array, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]

        has_vehicles = False
        valid_boxes = []
        total_detections = 0
        filtered_detections = {"too_small": 0, "wrong_class": 0, "low_conf": 0}
        vehicle_types = []
        confidences = []

        # Check if any vehicle classes detected with proper filtering
        if results.boxes is not None and len(results.boxes) > 0:
            total_detections = len(results.boxes)

            # Filter to only vehicle detections with proper size and confidence
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Only process vehicle classes
                if class_id not in VEHICLE_CLASSES:
                    filtered_detections["wrong_class"] += 1
                    continue

                # Check confidence threshold
                if confidence < CONFIDENCE_THRESHOLD:
                    filtered_detections["low_conf"] += 1
                    continue

                # Get box dimensions (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_width = x2 - x1
                box_height = y2 - y1

                # Filter out tiny detections (likely false positives)
                if box_width < MIN_BOX_SIZE or box_height < MIN_BOX_SIZE:
                    filtered_detections["too_small"] += 1
                    continue

                # This is a valid vehicle detection
                has_vehicles = True
                valid_boxes.append(box)

                # Track vehicle type
                vehicle_type_map = {2: 'car', 5: 'bus', 7: 'truck', 3: 'motorcycle', 1: 'bicycle', 0: 'person'}
                vehicle_types.append(vehicle_type_map.get(class_id, f'class_{class_id}'))
                confidences.append(confidence)

            # DEBUG: Log detections
            if total_detections > 0 and feed_id:
                import random
                if random.random() < 0.01:  # Log 1% of detections to avoid spam
                    print(f"  [YOLO] {feed_id}: {total_detections} total, {len(valid_boxes)} vehicles, filtered: {filtered_detections}")

            # Update tracker if enabled
            tracked_vehicles = []
            track_counts = {}
            if has_vehicles and tracker_manager and feed_id:
                confirmed_tracks, track_counts = tracker_manager.update_tracker(feed_id, results)
                tracked_vehicles = [
                    {
                        "track_id": track.track_id,
                        "class": track.class_name,
                        "confidence": round(track.confidence, 2),
                        "bbox": track.bbox.tolist()
                    }
                    for track in confirmed_tracks
                ]

            # Prepare detection data
            detection_data = {
                "vehicle_count": len(valid_boxes),
                "vehicle_types": list(set(vehicle_types)),  # Unique types
                "confidence_avg": round(sum(confidences) / len(confidences), 2) if confidences else 0,
                "tracked_vehicles": tracked_vehicles,
                "track_counts": track_counts
            }

            if has_vehicles:
                # Get image with drawn boxes using YOLO's built-in plotting
                annotated = results.plot()

                # Draw track IDs if tracking enabled
                if tracked_vehicles:
                    for track in tracked_vehicles:
                        x1, y1, x2, y2 = track["bbox"]
                        cv2.putText(
                            annotated,
                            f"ID:{track['track_id']}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )

                # OPTIMIZATION: Use OpenCV for faster encoding
                _, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                return True, encoded.tobytes(), detection_data

        return False, img_bytes, {}
    except Exception as e:
        print(f"Error in vehicle detection: {e}")
        import traceback
        traceback.print_exc()
        return False, img_bytes, {}


async def fetch_snapshot(feed: Dict) -> Optional[bytes]:
    """Fetch a single snapshot from a feed"""
    url = feed['imageUrl']

    async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
        try:
            response = await client.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()

            # Verify it's a valid image
            img_data = response.content
            try:
                Image.open(BytesIO(img_data)).verify()
                return img_data
            except:
                return None

        except Exception as e:
            return None


async def update_feed_cache_worker():
    """Background worker - processes feeds in parallel batches (OPTIMIZED)"""
    global cycle_counter
    print("Starting optimized feed cache worker...")

    # Get performance settings from config
    config_batch_size = CONFIG.get('performance', {}).get('batch_size', 240)
    http_config = CONFIG.get('performance', {}).get('http', {})

    # OPTIMIZATION: Dynamic batch size based on feed count, capped by config
    BATCH_SIZE = min(config_batch_size, max(100, len(feeds_data) // 10))

    # OPTIMIZATION: Connection pool from config
    limits = httpx.Limits(
        max_keepalive_connections=http_config.get('max_keepalive', 250),
        max_connections=http_config.get('max_connections', 300),
        keepalive_expiry=http_config.get('keepalive_expiry', 60)
    )
    timeout = httpx.Timeout(
        http_config.get('timeout', 5.0),
        connect=http_config.get('connect_timeout', 2.0)
    )

    async with httpx.AsyncClient(verify=False, timeout=timeout, limits=limits) as client:
        while True:
            if not feeds_data:
                await asyncio.sleep(5)
                continue

            cycle_counter += 1
            start_time = time.time()
            print(f"Starting cycle #{cycle_counter} - {len(feeds_data)} feeds (batch size: {BATCH_SIZE})...")

            # Fetch snapshot using shared client with exponential backoff
            async def fetch_with_client(feed):
                feed_id = feed['id']
                url = feed['imageUrl']

                # OPTIMIZATION: Skip feeds in backoff period
                if feed_id in feed_retry_after:
                    if time.time() < feed_retry_after[feed_id]:
                        return feed_id, None, False, "In backoff"

                try:
                    response = await client.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                    response.raise_for_status()
                    img_data = response.content
                    if len(img_data) > 1000:
                        # Clear backoff on success
                        if feed_id in feed_retry_after:
                            del feed_retry_after[feed_id]
                        return feed_id, img_data, True, None
                    return feed_id, None, False, "Image too small"
                except httpx.TimeoutException as e:
                    return feed_id, None, False, f"Timeout: {str(e)}"
                except httpx.ConnectError as e:
                    return feed_id, None, False, f"Connection error: {str(e)}"
                except Exception as e:
                    return feed_id, None, False, f"Error: {type(e).__name__}"

            # Process feeds in sequential batches
            batches = [feeds_data[i:i+BATCH_SIZE] for i in range(0, len(feeds_data), BATCH_SIZE)]

            all_results = []
            error_counts = {}
            detection_stats = {"skipped_unchanged": 0, "skipped_selective": 0, "processed": 0}

            for batch_num, batch in enumerate(batches, 1):
                batch_start = time.time()
                tasks = [fetch_with_client(feed) for feed in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                all_results.extend(results)

                # Get event loop for executor
                loop = asyncio.get_event_loop()

                # Update cache immediately after each batch completes
                for result in results:
                    if isinstance(result, tuple):
                        feed_id, img_data, success, error = result
                        if success and img_data:
                            # OPTIMIZATION: Check if image changed using hash
                            img_hash = hashlib.md5(img_data).hexdigest()
                            previous_hash = feed_image_hash.get(feed_id)

                            should_detect = True

                            if previous_hash == img_hash:
                                # Image unchanged, reuse previous detection
                                has_vehicles = feed_vehicle_detected.get(feed_id, False)
                                annotated_img = feed_cache.get(feed_id, img_data)
                                should_detect = False
                                detection_stats["skipped_unchanged"] += 1
                            elif cycle_counter > 1 and not feed_vehicle_detected.get(feed_id, False) and cycle_counter % SELECTIVE_SKIP_INTERVAL != 0:
                                # OPTIMIZATION: Selective detection - only re-check "empty" feeds every 2 cycles
                                # BUT: Always process on first cycle to establish baseline
                                has_vehicles = False
                                annotated_img = img_data
                                should_detect = False
                                detection_stats["skipped_selective"] += 1

                            if should_detect:
                                # OPTIMIZATION: Run YOLO in thread pool to avoid blocking event loop
                                has_vehicles, annotated_img, detection_data = await loop.run_in_executor(
                                    executor, detect_vehicles, img_data, feed_id
                                )
                                detection_stats["processed"] += 1

                                # Save detection to database if enabled and vehicles detected
                                if has_vehicles and db_manager and detection_data:
                                    try:
                                        await db_manager.add_detection(
                                            feed_id=feed_id,
                                            vehicle_count=detection_data.get("vehicle_count", 0),
                                            vehicle_types=detection_data.get("vehicle_types", []),
                                            confidence_avg=detection_data.get("confidence_avg", 0)
                                        )
                                    except Exception as e:
                                        print(f"Error saving detection to database: {e}")

                                # Send WebSocket update if vehicles detected
                                if has_vehicles and ws_manager and detection_data:
                                    try:
                                        await ws_manager.send_detection_update(feed_id, detection_data)
                                    except Exception as e:
                                        print(f"Error sending WebSocket update: {e}")

                            # Update hash
                            feed_image_hash[feed_id] = img_hash

                            # Store the annotated image (with boxes) in cache
                            feed_cache[feed_id] = annotated_img
                            feed_status[feed_id] = True
                            feed_vehicle_detected[feed_id] = has_vehicles

                            # Send streaming updates if enabled and vehicle detected
                            if has_vehicles and stream_config["enabled"]:
                                feed_info = next((f for f in feeds_data if f['id'] == feed_id), None)
                                if feed_info:
                                    try:
                                        if stream_config["format"] == "cot":
                                            cot_msg = generate_cot_message(feed_info)
                                            send_cot_udp(cot_msg, stream_config["ip"], stream_config["port"])
                                        elif stream_config["format"] == "lattice":
                                            publish_lattice_entity(feed_info, stream_config["latticeIntegration"], annotated_img)
                                    except Exception as e:
                                        print(f"Error streaming {stream_config['format']} for {feed_id}: {e}")
                        else:
                            feed_status[feed_id] = False
                            feed_vehicle_detected[feed_id] = False

                            # OPTIMIZATION: Exponential backoff for failed feeds
                            if error and "In backoff" not in error:
                                current_backoff = feed_retry_after.get(feed_id + "_interval", 15)
                                next_backoff = min(current_backoff * 2, 300)  # Max 5 minutes
                                feed_retry_after[feed_id] = time.time() + current_backoff
                                feed_retry_after[feed_id + "_interval"] = next_backoff

                            if error:
                                error_type = error.split(':')[0]
                                error_counts[error_type] = error_counts.get(error_type, 0) + 1

                elapsed = time.time() - batch_start
                success = sum(1 for r in results if isinstance(r, tuple) and r[2])
                print(f"    Batch {batch_num}/{len(batches)} done in {elapsed:.2f}s ({success}/{len(batch)} succeeded)")

            elapsed = time.time() - start_time
            working_count = sum(1 for status in feed_status.values() if status)
            vehicles_detected = sum(1 for v in feed_vehicle_detected.values() if v)
            print(f"Cycle #{cycle_counter} complete in {elapsed:.2f}s. {working_count}/{len(feeds_data)} working ({working_count/len(feeds_data)*100:.1f}%), {vehicles_detected} with vehicles")
            print(f"  Detection stats: {detection_stats['processed']} processed, {detection_stats['skipped_unchanged']} unchanged, {detection_stats['skipped_selective']} selective skip")

            # Show error breakdown
            if error_counts:
                print(f"  Error breakdown: {error_counts}")

            # Wait before next full update
            wait_time = max(0.5, 2.0 - elapsed)
            await asyncio.sleep(wait_time)


async def initialize_feeds():
    """Initialize feeds in background with prioritization"""
    global feeds_data

    # Fetch initial feed list
    await fetch_feed_list()

    # OPTIMIZATION: Prioritize major highways (國道) for faster initial cache warm-up
    if feeds_data:
        priority_feeds = [f for f in feeds_data if '國道' in f.get('roadName', '')]
        other_feeds = [f for f in feeds_data if '國道' not in f.get('roadName', '')]
        feeds_data = priority_feeds + other_feeds
        print(f"Feed prioritization: {len(priority_feeds)} priority feeds, {len(other_feeds)} others")

    # Sync feeds to database if enabled
    if db_manager and feeds_data:
        print(f"Syncing {len(feeds_data)} feeds to database...")
        for feed in feeds_data[:100]:  # Sync first 100 to avoid blocking
            try:
                await db_manager.upsert_feed(feed)
            except Exception as e:
                print(f"Error syncing feed {feed.get('id')}: {e}")
        print("Feed sync complete")

    # Start background cache worker
    asyncio.create_task(update_feed_cache_worker())

    # Refresh feed list every hour
    async def refresh_feed_list():
        while True:
            await asyncio.sleep(3600)  # 1 hour
            await fetch_feed_list()

            # Sync new feeds to database
            if db_manager and feeds_data:
                for feed in feeds_data[:50]:  # Sync subset on refresh
                    try:
                        await db_manager.upsert_feed(feed)
                    except Exception:
                        pass

    asyncio.create_task(refresh_feed_list())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global yolo_model, executor, db_manager, tracker_manager, health_checker
    import os

    # Create application state object for health checker
    class AppState:
        def __init__(self):
            self.feeds_data = feeds_data
            self.feed_cache = feed_cache
            self.feed_status = feed_status
            self.yolo_model = None
            self.http_client = None  # HTTP client is created on-demand in fetch operations
            self.ws_manager = ws_manager
            self.db_manager = None
            self.tracker_manager = None

    app_state = AppState()
    app.state.cctv = app_state

    # Initialize health checker
    health_checker = HealthChecker(app_state)

    # Startup
    logger.info("Starting Taiwan CCTV API (OPTIMIZED)...")
    logger.info("Loading YOLOv8n model...")

    # Load YOLO model (downloads automatically if not present)
    try:
        yolo_model = YOLO('yolov8n.pt')  # Nano model for speed
        app_state.yolo_model = yolo_model
        logger.info("YOLOv8n model loaded successfully")

        # Set system info metrics
        metrics.system_info.info({
            'version': '1.0.0',
            'model': 'yolov8n',
            'python_version': str(sys.version_info[:3])
        })
    except Exception as e:
        logger.error("Could not load YOLO model", error=str(e))
        logger.warning("Vehicle detection will be disabled")
        await alert_manager.send_alert(
            alert_type="startup_failure",
            severity="critical",
            message="Failed to load YOLO model",
            metadata={"error": str(e)}
        )

    # OPTIMIZATION: Initialize ThreadPoolExecutor for YOLO inference
    # Use config value or CPU count
    max_workers = CONFIG.get('performance', {}).get('worker_threads', min(os.cpu_count() or 4, 8))
    executor = ThreadPoolExecutor(max_workers=max_workers)
    logger.info("ThreadPoolExecutor initialized", max_workers=max_workers)

    # Initialize database if enabled
    db_config = CONFIG.get('database', {})
    if db_config.get('enabled', True):
        db_type = db_config.get('type', 'sqlite')
        if db_type == 'sqlite':
            db_path = db_config.get('path', './data/cctv_data.db')
            # Ensure data directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite+aiosqlite:///{db_path}"
        else:
            # PostgreSQL/MySQL configuration
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 5432)
            username = db_config.get('username', 'user')
            password = db_config.get('password', 'pass')
            database = db_config.get('database', 'cctv')
            database_url = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"

        try:
            db_manager = DatabaseManager(database_url)
            await db_manager.init_db()
            app_state.db_manager = db_manager
            logger.info("Database initialized", db_type=db_type)

            # Schedule periodic cleanup
            async def cleanup_task():
                while True:
                    await asyncio.sleep(86400)  # Once per day
                    retention = db_config.get('retention', {})
                    await db_manager.cleanup_old_data(
                        detection_retention_days=retention.get('detections', 30),
                        history_retention_days=retention.get('feed_history', 90)
                    )

            asyncio.create_task(cleanup_task())
        except Exception as e:
            logger.error("Could not initialize database", error=str(e))
            db_manager = None
            await alert_manager.send_alert(
                alert_type="database_init_failure",
                severity="error",
                message="Failed to initialize database",
                metadata={"error": str(e)}
            )
    else:
        logger.info("Database disabled in config")

    # Initialize tracker if enabled
    tracking_config = CONFIG.get('detection', {}).get('tracking', {})
    if tracking_config.get('enabled', True):
        tracker_manager = TrackerManager(
            max_age=tracking_config.get('max_age', 30),
            min_hits=tracking_config.get('min_hits', 3),
            iou_threshold=tracking_config.get('iou_threshold', 0.3)
        )
        app_state.tracker_manager = tracker_manager
        logger.info("Vehicle tracking initialized")
    else:
        logger.info("Vehicle tracking disabled in config")

    # Start background tasks
    asyncio.create_task(initialize_feeds())

    # WebSocket heartbeat task
    websocket_config = CONFIG.get('websocket', {})
    if websocket_config.get('enabled', True):
        async def heartbeat_task():
            interval = websocket_config.get('heartbeat_interval', 30)
            while True:
                await asyncio.sleep(interval)
                await ws_manager.send_heartbeat()

        asyncio.create_task(heartbeat_task())
        logger.info("WebSocket heartbeat started", interval=websocket_config.get('heartbeat_interval', 30))

    # Metrics update task
    async def metrics_update_task():
        """Periodically update Prometheus metrics"""
        while True:
            await asyncio.sleep(10)  # Update every 10 seconds
            try:
                metrics.feeds_total.set(len(feeds_data))
                metrics.feeds_online.set(sum(1 for status in feed_status.values() if status))
                metrics.feeds_with_vehicles.set(sum(1 for v in feed_vehicle_detected.values() if v))
                metrics.cache_size_bytes.set(sum(len(v) for v in feed_cache.values()))
                metrics.active_websockets.set(len(ws_manager.active_connections))
            except Exception as e:
                logger.error("Error updating metrics", error=str(e))

    asyncio.create_task(metrics_update_task())
    logger.info("Metrics update task started")

    yield

    # Shutdown (cleanup if needed)
    logger.info("Shutting down Taiwan CCTV API...")
    if executor:
        executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor shut down")
    if db_manager:
        await db_manager.close()
        logger.info("Database connection closed")


app = FastAPI(title="Taiwan CCTV API (Optimized)", lifespan=lifespan)

# OPTIMIZATION: Add GZip compression middleware (30-50% smaller payloads)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
async def root():
    """API root"""
    return {
        "name": "Taiwan CCTV API",
        "version": "1.0",
        "feeds": len(feeds_data),
        "cached": len(feed_cache),
        "working": sum(1 for status in feed_status.values() if status)
    }


@app.get("/api/feeds")
async def get_feeds():
    """Get all feed metadata"""
    return {
        "feeds": feeds_data,
        "status": feed_status,
        "vehicleDetected": feed_vehicle_detected,
        "lastUpdate": last_update
    }


@app.get("/api/feeds/{feed_id}/snapshot")
async def get_snapshot(feed_id: str):
    """Get latest cached snapshot for a feed"""
    if feed_id not in feed_cache:
        raise HTTPException(status_code=404, detail="Feed not found or offline")

    return Response(content=feed_cache[feed_id], media_type="image/jpeg")


@app.get("/api/feeds/{feed_id}/stream")
async def get_stream(feed_id: str):
    """Get MJPEG stream for a feed"""
    # Find feed
    feed = next((f for f in feeds_data if f['id'] == feed_id), None)
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    url = feed['streamUrl']

    async def stream_generator():
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            try:
                async with client.stream('GET', url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        yield chunk
            except Exception as e:
                print(f"Stream error for {feed_id}: {e}")

    return StreamingResponse(
        stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=--DIGIEVER"
    )


@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint for monitoring and alerting.
    Returns detailed status of all system components.
    """
    if health_checker is None:
        return {
            "status": "unhealthy",
            "message": "Health checker not initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    try:
        health_status = await health_checker.check_all()
        return health_status
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "message": f"Health check error: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@app.get("/health/live")
async def liveness_probe():
    """
    Liveness probe for Kubernetes/container orchestration.
    Returns 200 if the application is running.
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health/ready")
async def readiness_probe():
    """
    Readiness probe for Kubernetes/container orchestration.
    Returns 200 if the application is ready to serve traffic.
    """
    # Check critical components
    is_ready = (
        yolo_model is not None and
        len(feeds_data) > 0 and
        (db_manager is not None or not CONFIG.get('database', {}).get('enabled', True))
    )

    if is_ready:
        return {
            "status": "ready",
            "feeds_loaded": len(feeds_data),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    else:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "yolo_loaded": yolo_model is not None,
                "feeds_loaded": len(feeds_data),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    working = sum(1 for status in feed_status.values() if status)
    vehicles_detected = sum(1 for v in feed_vehicle_detected.values() if v)

    stats = {
        "totalFeeds": len(feeds_data),
        "cachedFeeds": len(feed_cache),
        "workingFeeds": working,
        "offlineFeeds": len(feed_status) - working,
        "vehiclesDetectedFeeds": vehicles_detected,
        "lastUpdate": last_update,
        "cacheSize": sum(len(img) for img in feed_cache.values()) / (1024 * 1024)  # MB
    }

    # Add WebSocket stats if available
    if ws_manager:
        stats["websocket"] = ws_manager.get_stats()

    return stats


@app.get("/api/operational/status")
async def get_operational_status():
    """
    Operational status dashboard endpoint.
    Provides comprehensive system metrics for monitoring and troubleshooting.
    """
    working = sum(1 for status in feed_status.values() if status)
    vehicles_detected = sum(1 for v in feed_vehicle_detected.values() if v)
    cache_size_mb = sum(len(img) for img in feed_cache.values()) / (1024 * 1024)

    # Calculate uptime metrics
    import psutil
    process = psutil.Process()
    uptime_seconds = time.time() - process.create_time()

    operational_status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": int(uptime_seconds),
        "uptime_human": f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m",

        "feeds": {
            "total": len(feeds_data),
            "online": working,
            "offline": len(feed_status) - working,
            "online_percentage": round((working / len(feeds_data) * 100) if feeds_data else 0, 2),
            "with_vehicles": vehicles_detected,
            "last_update": last_update
        },

        "cache": {
            "size_mb": round(cache_size_mb, 2),
            "items": len(feed_cache),
            "avg_size_kb": round((cache_size_mb * 1024 / len(feed_cache)) if feed_cache else 0, 2)
        },

        "components": {
            "yolo_model": "loaded" if yolo_model else "not_loaded",
            "database": "enabled" if db_manager else "disabled",
            "tracker": "enabled" if tracker_manager else "disabled",
            "websocket": "enabled" if ws_manager else "disabled"
        },

        "websocket": {
            "active_connections": len(ws_manager.active_connections) if ws_manager else 0,
            "stats": ws_manager.get_stats() if ws_manager else {}
        },

        "circuit_breakers": {
            "feed_fetcher": feed_circuit_breaker.get_state()
        },

        "alerts": {
            "recent": alert_manager.get_recent_alerts(limit=10)
        }
    }

    # Add database stats if available
    if db_manager:
        try:
            from database import Detection as DetectionModel, VehicleTrack as VehicleTrackModel
            from sqlalchemy import func, select
            async with db_manager.session() as session:
                detection_count = await session.execute(
                    select(func.count()).select_from(DetectionModel)
                )
                track_count = await session.execute(
                    select(func.count()).select_from(VehicleTrackModel)
                )

                operational_status["database"] = {
                    "detections_count": detection_count.scalar(),
                    "tracks_count": track_count.scalar()
                }
        except Exception as e:
            logger.error("Failed to get database stats", error=str(e))
            operational_status["database"] = {"error": str(e)}

    return operational_status


@app.get("/api/feeds/{feed_id}/history")
async def get_feed_history(feed_id: str, hours: int = 24):
    """Get detection history for a specific feed"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not enabled")

    try:
        detections = await db_manager.get_feed_history(feed_id, hours)
        return {
            "feed_id": feed_id,
            "hours": hours,
            "detections": [
                {
                    "id": d.id,
                    "timestamp": d.timestamp.isoformat(),
                    "vehicle_count": d.vehicle_count,
                    "vehicle_types": d.vehicle_types,
                    "confidence_avg": d.confidence_avg
                }
                for d in detections
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


@app.get("/api/feeds/{feed_id}/stats")
async def get_feed_stats(feed_id: str, days: int = 7):
    """Get statistics for a specific feed"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not enabled")

    try:
        stats = await db_manager.get_feed_stats(feed_id, days)
        return {
            "feed_id": feed_id,
            **stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


@app.get("/api/search")
async def search_feeds(
    road: Optional[str] = None,
    has_vehicles: Optional[bool] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None
):
    """Search feeds with filters"""
    results = feeds_data.copy()

    # Filter by road name
    if road:
        results = [f for f in results if road.lower() in f.get('roadName', '').lower()]

    # Filter by vehicle detection
    if has_vehicles is not None:
        results = [
            f for f in results
            if feed_vehicle_detected.get(f['id'], False) == has_vehicles
        ]

    # Filter by bounding box
    if lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
        results = [
            f for f in results
            if (lat_min <= float(f.get('lat', 0) or 0) <= lat_max and
                lon_min <= float(f.get('lon', 0) or 0) <= lon_max)
        ]

    # Add status and vehicle detection to results
    enriched_results = []
    for feed in results:
        feed_copy = feed.copy()
        feed_copy['status'] = feed_status.get(feed['id'], False)
        feed_copy['vehicleDetected'] = feed_vehicle_detected.get(feed['id'], False)
        enriched_results.append(feed_copy)

    return {
        "query": {
            "road": road,
            "has_vehicles": has_vehicles,
            "bbox": [lat_min, lon_min, lat_max, lon_max] if all([lat_min, lat_max, lon_min, lon_max]) else None
        },
        "count": len(enriched_results),
        "feeds": enriched_results
    }


@app.get("/api/map")
async def get_map_data():
    """Get feed data formatted for map display"""
    map_features = []

    for feed in feeds_data:
        lat = float(feed.get('lat', 0) or 0)
        lon = float(feed.get('lon', 0) or 0)

        if lat == 0 or lon == 0:
            continue

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "id": feed['id'],
                "roadName": feed.get('roadName', ''),
                "locationMile": feed.get('locationMile', ''),
                "description": feed.get('description', ''),
                "direction": feed.get('direction', ''),
                "isWorking": feed_status.get(feed['id'], False),
                "hasVehicles": feed_vehicle_detected.get(feed['id'], False)
            }
        }
        map_features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": map_features
    }


@app.get("/api/stream/config")
async def get_stream_config():
    """Get stream out configuration"""
    return stream_config


@app.post("/api/stream/config")
async def set_stream_config(config: StreamConfig):
    """Set stream out configuration"""
    global stream_config
    stream_config["enabled"] = config.enabled
    stream_config["format"] = config.format
    stream_config["ip"] = config.ip
    stream_config["port"] = config.port
    stream_config["latticeToken"] = config.latticeToken or ""
    stream_config["latticeSandboxToken"] = config.latticeSandboxToken or ""
    stream_config["latticeIntegration"] = config.latticeIntegration or "taiwan-cctv"
    stream_config["latticeUrl"] = config.latticeUrl or ""

    # Initialize Lattice client if token provided and format is lattice
    if config.format == "lattice" and config.latticeToken:
        initialize_lattice_client(
            env_token=config.latticeToken,
            sandbox_token=config.latticeSandboxToken or None,
            base_url=config.latticeUrl or None
        )

    status = "enabled" if config.enabled else "disabled"
    if config.format == "cot":
        print(f"Stream Out {status}: CoT -> {config.ip}:{config.port}")
    elif config.format == "lattice":
        print(f"Stream Out {status}: Lattice -> {config.latticeIntegration} @ {config.latticeUrl or 'default'}")

    return {"status": "success", "config": stream_config}


@app.post("/api/database/reset")
async def reset_database():
    """Reset database - clear all detections and tracks"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not enabled")

    try:
        from database import Detection as DetectionModel, VehicleTrack as VehicleTrackModel
        from sqlalchemy import delete

        async with db_manager.session() as session:
            # Delete all vehicle tracks
            await session.execute(delete(VehicleTrackModel))
            # Delete all detections
            await session.execute(delete(DetectionModel))
            await session.commit()

        logger.info("Database reset completed - all detections and tracks cleared")

        await alert_manager.send_alert(
            alert_type="database_reset",
            severity="warning",
            message="Database has been manually reset",
            metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
        )

        return {
            "status": "success",
            "message": "Database reset completed",
            "cleared": ["detections", "vehicle_tracks"]
        }

    except Exception as e:
        logger.error("Failed to reset database", error=str(e))
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    client_id = websocket.query_params.get("client_id", "anonymous")
    await ws_manager.connect(websocket, client_id)

    try:
        # Handle incoming messages
        await handle_websocket_messages(websocket, ws_manager)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

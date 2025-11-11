#!/usr/bin/env python3
"""
FastAPI backend for Taiwan CCTV Viewer
Ingests and caches all 2,500+ feeds in real-time
"""

import asyncio
import ssl
import time
from typing import Dict, List, Optional
from io import BytesIO
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import numpy as np

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from ultralytics import YOLO

# Global state
feeds_data: List[Dict] = []
feed_cache: Dict[str, bytes] = {}  # feed_id -> latest JPEG bytes
feed_status: Dict[str, bool] = {}  # feed_id -> is_working
feed_vehicle_detected: Dict[str, bool] = {}  # feed_id -> has_vehicles
last_update: float = 0

# YOLO model (loaded at startup)
yolo_model = None
# Vehicle class IDs in COCO dataset: car=2, bus=5, truck=7 (excluding motorcycle to reduce false positives)
VEHICLE_CLASSES = [2, 5, 7]
# Minimum detection box size (width or height) to filter out tiny false positives
MIN_BOX_SIZE = 30  # pixels
# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.6

# SSL context for Taiwan servers
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


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


def detect_vehicles(img_bytes: bytes) -> tuple[bool, bytes]:
    """Detect if vehicles are present in image using YOLO
    Returns: (has_vehicles, annotated_image_bytes)
    """
    try:
        if yolo_model is None:
            return False, img_bytes

        # Convert bytes to PIL Image
        img = Image.open(BytesIO(img_bytes))
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]

        # Run YOLO inference with confidence threshold
        results = yolo_model(img_array, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]

        has_vehicles = False
        valid_boxes = []

        # Check if any vehicle classes detected with proper filtering
        if results.boxes is not None and len(results.boxes) > 0:
            # Filter to only vehicle detections with proper size and confidence
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Only process vehicle classes
                if class_id not in VEHICLE_CLASSES:
                    continue

                # Check confidence threshold
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                # Get box dimensions (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_width = x2 - x1
                box_height = y2 - y1

                # Filter out tiny detections (likely false positives)
                if box_width < MIN_BOX_SIZE or box_height < MIN_BOX_SIZE:
                    continue

                # This is a valid vehicle detection
                has_vehicles = True
                valid_boxes.append(box)

            if has_vehicles:
                # Get image with drawn boxes using YOLO's built-in plotting
                # This will only show boxes that passed all our filters
                annotated = results.plot()  # Returns numpy array with boxes drawn

                # Convert annotated numpy array back to JPEG bytes
                annotated_img = Image.fromarray(annotated)
                output = BytesIO()
                annotated_img.save(output, format='JPEG', quality=85)
                return True, output.getvalue()

        return False, img_bytes
    except Exception as e:
        print(f"Error in vehicle detection: {e}")
        return False, img_bytes


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
    """Background worker - processes feeds in parallel batches"""
    print("Starting feed cache worker...")

    # Create persistent HTTP client with reasonable limits
    limits = httpx.Limits(max_keepalive_connections=100, max_connections=200, keepalive_expiry=30)
    timeout = httpx.Timeout(5.0, connect=2.0)  # Feeds respond in ~2s when not overwhelmed

    async with httpx.AsyncClient(verify=False, timeout=timeout, limits=limits) as client:
        while True:
            if not feeds_data:
                await asyncio.sleep(5)
                continue

            start_time = time.time()
            print(f"Starting concurrent update of {len(feeds_data)} feeds...")

            # Fetch snapshot using shared client
            async def fetch_with_client(feed):
                url = feed['imageUrl']
                try:
                    response = await client.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                    response.raise_for_status()
                    img_data = response.content
                    if len(img_data) > 1000:
                        return feed['id'], img_data, True, None
                    return feed['id'], None, False, "Image too small"
                except httpx.TimeoutException as e:
                    return feed['id'], None, False, f"Timeout: {str(e)}"
                except httpx.ConnectError as e:
                    return feed['id'], None, False, f"Connection error: {str(e)}"
                except Exception as e:
                    return feed['id'], None, False, f"Error: {type(e).__name__}"

            # Process feeds in sequential batches to avoid overwhelming servers
            BATCH_SIZE = 200  # Smaller batches, processed one at a time
            batches = [feeds_data[i:i+BATCH_SIZE] for i in range(0, len(feeds_data), BATCH_SIZE)]

            print(f"  Processing {len(batches)} batches sequentially...")

            all_results = []
            error_counts = {}

            for batch_num, batch in enumerate(batches, 1):
                batch_start = time.time()
                tasks = [fetch_with_client(feed) for feed in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                all_results.extend(results)

                # Update cache immediately after each batch completes
                for result in results:
                    if isinstance(result, tuple):
                        feed_id, img_data, success, error = result
                        if success and img_data:
                            # Run YOLO vehicle detection and get annotated image
                            has_vehicles, annotated_img = detect_vehicles(img_data)

                            # Store the annotated image (with boxes) in cache
                            feed_cache[feed_id] = annotated_img
                            feed_status[feed_id] = True
                            feed_vehicle_detected[feed_id] = has_vehicles
                        else:
                            feed_status[feed_id] = False
                            feed_vehicle_detected[feed_id] = False
                            if error:
                                error_type = error.split(':')[0]
                                error_counts[error_type] = error_counts.get(error_type, 0) + 1

                elapsed = time.time() - batch_start
                success = sum(1 for r in results if isinstance(r, tuple) and r[2])
                print(f"    Batch {batch_num}/{len(batches)} done in {elapsed:.2f}s ({success}/{len(batch)} succeeded)")


            elapsed = time.time() - start_time
            working_count = sum(1 for status in feed_status.values() if status)
            print(f"Cache update complete in {elapsed:.2f}s. {working_count}/{len(feeds_data)} feeds working ({working_count/len(feeds_data)*100:.1f}%)")

            # Show error breakdown
            if error_counts:
                print(f"  Error breakdown: {error_counts}")

            # Wait before next full update
            wait_time = max(0.5, 2.0 - elapsed)
            await asyncio.sleep(wait_time)


async def initialize_feeds():
    """Initialize feeds in background"""
    # Fetch initial feed list
    await fetch_feed_list()

    # Start background cache worker
    asyncio.create_task(update_feed_cache_worker())

    # Refresh feed list every hour
    async def refresh_feed_list():
        while True:
            await asyncio.sleep(3600)  # 1 hour
            await fetch_feed_list()

    asyncio.create_task(refresh_feed_list())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global yolo_model

    # Startup
    print("Starting Taiwan CCTV API...")
    print("Loading YOLOv8n model...")

    # Load YOLO model (downloads automatically if not present)
    try:
        yolo_model = YOLO('yolov8n.pt')  # Nano model for speed
        print("YOLOv8n model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load YOLO model: {e}")
        print("Vehicle detection will be disabled")

    # Start background tasks
    asyncio.create_task(initialize_feeds())

    yield

    # Shutdown (cleanup if needed)
    print("Shutting down Taiwan CCTV API...")


app = FastAPI(title="Taiwan CCTV API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    working = sum(1 for status in feed_status.values() if status)
    return {
        "totalFeeds": len(feeds_data),
        "cachedFeeds": len(feed_cache),
        "workingFeeds": working,
        "offlineFeeds": len(feed_status) - working,
        "lastUpdate": last_update,
        "cacheSize": sum(len(img) for img in feed_cache.values()) / (1024 * 1024)  # MB
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

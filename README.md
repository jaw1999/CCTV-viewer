# Taiwan Highway CCTV Viewer with Real-Time Vehicle Detection

Real-time viewer for 2,500+ Taiwan Highway Bureau CCTV feeds with YOLOv8 vehicle detection, tracking, database storage, WebSocket updates, and Stream Out integration.

## Features

### Backend
- **YOLOv8 Vehicle Detection**: Cars, buses, trucks at 60% confidence with bounding boxes
- **Object Tracking**: Multi-object tracking with unique IDs and vehicle counting
- **Database**: SQLite storage for detections, tracks, and historical data
- **WebSocket**: Real-time push notifications for vehicle detections
- **Stream Out Integration**:
  - CoT (Cursor on Target) via UDP for TAK/ATAK
  - Lattice (Anduril) via REST API
- **Performance Optimizations**:
  - Concurrent ingestion with connection pooling
  - Selective YOLO processing (skip unchanged frames)
  - Async operations throughout
  - In-memory caching
- **REST API**: Comprehensive endpoints for feeds, stats, search, and map data

### Frontend
- **Grid View** ([index.html](client/index.html))
  - Real-time vehicle detection badges ("TAI OCCUPIED")
  - Auto-refresh thumbnails (2 second interval)
  - Filter by status (All/Working Only/Vehicles Detected)
  - Search by location/road/ID
  - Stream Out configuration UI
  - WebSocket real-time updates

- **Map View** ([map.html](client/map.html))
  - Interactive Leaflet map with all camera locations
  - Color-coded markers (green=online, red=vehicles, gray=offline)
  - Marker clustering for performance
  - Real-time marker updates via WebSocket
  - Filter by road name and vehicle detection
  - Click markers for live snapshots and details

- **Single Feed View** ([feed.html](client/feed.html))
  - High-frequency updates (500ms)
  - Full camera details and location
  - Vehicle detection indicators

## Architecture

```
┌─────────────────┐         ┌────────────────────────────┐         ┌─────────────────┐
│   Taiwan        │         │  FastAPI Backend (8001)    │         │   Web Client    │
│   Highway       │────────▶│  - YOLOv8n Detector        │◀───────▶│   (Browser)     │
│   Bureau        │  HTTPS  │  - Object Tracker          │  HTTP   │   Port 8000     │
│   (2500 feeds)  │         │  - SQLite Database         │  + WS   │                 │
└─────────────────┘         │  - WebSocket Manager       │         └─────────────────┘
                            │  - Memory Cache            │
                            └────────────────────────────┘
                                       │
                                       │ Concurrent Ingestion
                                       │ (Batches of 240)
                                       │
                                       ▼
                            ┌──────────────────┐
                            │   SQLite DB      │
                            │  - detections    │
                            │  - tracks        │
                            │  - feeds         │
                            └──────────────────┘
                                       │
                                       ▼
                            ┌──────────────────┐
                            │   Stream Out     │
                            │   (Optional)     │──UDP──▶ TAK/ATAK (CoT)
                            │                  │
                            │                  │──HTTPS─▶ Lattice (Anduril)
                            └──────────────────┘
```

## Quick Start

### Automatic (Recommended)
```bash
./start.sh
```

Then open: **http://localhost:8000**

The startup script will:
1. Create/activate virtual environment
2. Install all dependencies (including YOLOv8)
3. Start backend server on port 8001
4. Start frontend server on port 8000
5. Handle graceful shutdown with Ctrl+C

### Manual Setup

**Terminal 1 - Backend:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python backend/main.py
```

**Terminal 2 - Frontend:**
```bash
cd client
python3 -m http.server 8000
```

Then open: **http://localhost:8000**

## API Endpoints

### Core Endpoints

#### GET /api/feeds
Returns all feed metadata, online status, and vehicle detection status

**Response:**
```json
{
  "feeds": [
    {
      "id": "CCTV-14-0620-009-002",
      "streamUrl": "https://...",
      "imageUrl": "https://.../snapshot",
      "roadName": "台62線",
      "locationMile": "9K+020",
      "lat": "25.10529",
      "lon": "121.7321",
      "direction": "W",
      "description": "..."
    }
  ],
  "status": {
    "CCTV-14-0620-009-002": true
  },
  "vehicleDetected": {
    "CCTV-14-0620-009-002": true
  },
  "lastUpdate": 1699999999.0
}
```

#### GET /api/feeds/{feed_id}/snapshot
Returns cached JPEG snapshot for a specific feed (with YOLO bounding boxes if vehicles detected)

#### GET /api/feeds/{feed_id}/stream
Returns live MJPEG stream (for future use)

#### GET /api/stats
Returns system statistics including detection performance

**Response:**
```json
{
  "totalFeeds": 2402,
  "cachedFeeds": 2402,
  "workingFeeds": 1876,
  "offlineFeeds": 526,
  "vehiclesDetectedFeeds": 145,
  "lastUpdate": 1699999999.0,
  "cacheSize": 450.5,
  "detectionStats": {
    "processed": 1876,
    "skipped_unchanged": 430,
    "skipped_selective": 620
  },
  "websocket": {
    "connections": 2,
    "total_messages": 1234
  }
}
```

### Search & Map Endpoints

#### GET /api/search
Search feeds with filters

**Parameters:**
- `road` (optional): Road name filter
- `has_vehicles` (optional): true/false
- `lat_min`, `lat_max`, `lon_min`, `lon_max` (optional): Bounding box

**Response:**
```json
{
  "query": {
    "road": "國道1號",
    "has_vehicles": true
  },
  "count": 42,
  "feeds": [ /* enriched feed objects with status and vehicleDetected */ ]
}
```

#### GET /api/map
Get all feeds as GeoJSON for map display

**Response:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [121.7321, 25.10529]
      },
      "properties": {
        "id": "CCTV-14-0620-009-002",
        "roadName": "台62線",
        "isWorking": true,
        "hasVehicles": false
      }
    }
  ]
}
```

### Stream Out Endpoints

#### POST /api/stream/config
Configure Stream Out integration

**Request:**
```json
{
  "enabled": true,
  "format": "cot",
  "ip": "127.0.0.1",
  "port": 8087,
  "latticeUrl": "",
  "latticeToken": "",
  "latticeSandboxToken": "",
  "latticeIntegration": "taiwan-cctv"
}
```

#### GET /api/stream/config
Get current Stream Out configuration

### WebSocket Endpoint

#### WS /ws
Real-time updates for vehicle detections and feed status

**Client → Server Messages:**
```json
{"action": "subscribe", "feed_id": "CCTV-14-0620-009-002"}
{"action": "unsubscribe", "feed_id": "CCTV-14-0620-009-002"}
{"action": "ping"}
```

**Server → Client Messages:**
```json
// Vehicle detection
{
  "type": "detection",
  "feed_id": "CCTV-14-0620-009-002",
  "timestamp": "2025-11-12T21:30:45.123Z",
  "data": {
    "vehicle_count": 3,
    "vehicle_types": ["car", "car", "truck"],
    "tracked_vehicles": [...],
    "track_counts": {"car": 2, "truck": 1}
  }
}

// Feed status update
{
  "type": "feed_status",
  "feed_id": "CCTV-14-0620-009-002",
  "timestamp": "2025-11-12T21:30:45.123Z",
  "is_working": true,
  "has_vehicles": true
}

// Server heartbeat (every 30s)
{
  "type": "heartbeat",
  "timestamp": "2025-11-12T21:30:45.123Z",
  "connections": 2
}

// Stats broadcast
{
  "type": "stats",
  "timestamp": "2025-11-12T21:30:45.123Z",
  "data": { /* stats object */ }
}
```

## Stream Out Integration

### CoT (Cursor on Target)
Send vehicle detection events to TAK/ATAK systems via UDP.

**Config:** Format, IP, Port

**Message:** XML event with camera location, metadata, timestamp, video link

**Example:**
```xml
<event version="2.0" uid="TrafficCam-CCTV-14-0620-009-002" type="a-u-G" time="..." start="..." stale="...">
  <point lat="25.10529" lon="121.7321" hae="0" ce="50" le="0"/>
  <detail>
    <contact callsign="TrafficCam-CCTV-14-0620-009-002"/>
    <remarks>台62線 at 9K+020 - Vehicles detected</remarks>
    <link url="http://localhost:8001/api/feeds/CCTV-14-0620-009-002/snapshot"/>
  </detail>
</event>
```

### Lattice (Anduril)
Publish vehicle track entities to Lattice platform.

**Config:** Format, URL, Environment Token, Sandbox Token (for sandboxes), Integration Name

**Entity:** Camera ID, location, VEHICLE platform type, 1-hour expiry

**Note:** Sandboxes need two tokens (Authorization + anduril-sandbox-authorization headers)

## YOLO Vehicle Detection

- **Model**: YOLOv8n (nano - fast inference)
- **Classes**: Car, Bus, Truck (COCO dataset IDs: 2, 5, 7)
- **Confidence**: 60% minimum
- **Min box size**: 30px (filters false positives from distant objects)
- **Processing**: ~50-100ms per image
- **Optimizations**:
  - Skip detection on unchanged frames (hash-based)
  - Selective skip on empty feeds (every 2nd cycle)
  - Async processing via thread pool

## Object Tracking

- **Tracker**: Custom IoU-based tracker
- **Features**:
  - Unique track IDs for each vehicle
  - Vehicle counting per class
  - Track persistence across frames
  - Confirmed tracks (minimum 3 hits)
- **Storage**: Tracks saved to database with timestamps

## Database Schema

### Tables

**feeds**
- id, road_name, location_mile, lat, lon, direction, last_online

**detections**
- id, feed_id, timestamp, vehicle_count, detection_data (JSON)

**vehicle_tracks**
- id, feed_id, track_id, first_seen, last_seen, vehicle_class, speed_estimate

## Configuration

Edit [config.yaml](config.yaml) to customize:

```yaml
detection:
  enabled: true
  confidence_threshold: 0.6
  min_box_size: 30
  vehicle_classes: [2, 5, 7]  # car, bus, truck

database:
  enabled: true
  path: "./data/detections.db"

tracking:
  enabled: true
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

websocket:
  enabled: true
  heartbeat_interval: 30

performance:
  batch_size: 240
  max_connections: 100
  selective_skip_interval: 2
```

## Technical Details

- **Feeds**: 2,402 Taiwan highway camera feeds
- **Backend**: Python 3.8+, FastAPI, YOLOv8, SQLAlchemy, aiosqlite
- **Frontend**: Vanilla JavaScript, Leaflet.js for maps
- **Memory**: ~450 MB for cached JPEGs
- **Refresh Rate**: 2 seconds (grid), 0.5 seconds (single feed)
- **Detection Rate**: ~1876 feeds processed per cycle (~20-30 seconds)

## Troubleshooting

### Backend won't start
```bash
# Install dependencies manually
python3 -m pip install -r requirements.txt

# Run with verbose logging
python3 backend/main.py
```

### Feeds showing as offline
- Check backend console for errors
- Verify network connectivity to Taiwan
- Some feeds may actually be offline (normal - typically ~500/2400)
- Check `/api/stats` endpoint for system health

### YOLO not detecting vehicles
- Check backend logs for "YOLO model loaded successfully"
- Ensure PyTorch and dependencies installed correctly
- Detection requires confidence ≥0.6 (60%) and minimum box size of 30 pixels
- Only detects cars, buses, and trucks (motorcycles excluded)
- Some cameras may not have vehicles in frame

### WebSocket not connecting
- Check browser console for WebSocket errors
- Verify backend is running on port 8001
- Check firewall/proxy settings
- WebSocket URL: `ws://localhost:8001/ws`

### Slow refresh rate
- Backend logs show actual fetch time + YOLO processing time
- Network latency to Taiwan servers affects speed
- YOLO adds ~50-100ms per image
- Adjust `REFRESH_INTERVAL` in frontend code if needed
- Check `detectionStats` in `/api/stats` for performance metrics

### High memory usage
- Each cached JPEG is ~200 KB
- 2,400 feeds = ~480 MB baseline
- This is normal and expected
- Database grows over time (detections history)

### Map markers not updating
- Check browser console for "Map: WebSocket connected"
- Markers update in real-time when vehicles detected
- Try refreshing the page to reconnect WebSocket

## Project Structure

```
videoviewer/
├── backend/
│   ├── main.py                 # FastAPI server + YOLO + WebSocket
│   ├── database.py             # SQLAlchemy models and DB manager
│   ├── tracker.py              # Multi-object tracking
│   └── websocket_manager.py    # WebSocket connection handler
├── client/
│   ├── index.html              # Grid view with vehicle detection
│   ├── map.html                # Interactive map view
│   └── feed.html               # Single feed detailed view
├── data/
│   └── detections.db           # SQLite database (auto-created)
├── venv/                       # Python virtual environment
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── start.sh                    # Startup script
├── IMPLEMENTATION_ROADMAP.md   # Feature roadmap
├── OPTIMIZATIONS.md            # Performance optimizations guide
└── README.md                   # This file
```


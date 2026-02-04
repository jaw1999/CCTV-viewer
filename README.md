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
  - ChatSurfer (NRO) via REST API
- **Performance Optimizations**:
  - Concurrent ingestion with connection pooling
  - Selective YOLO processing (skip unchanged frames)
  - Async operations throughout
  - In-memory caching
- **REST API**: Comprehensive endpoints for feeds, stats, search, and map data
- **Operational Monitoring**:
  - Health check endpoints for all system components
  - Prometheus metrics export for performance monitoring
  - Structured JSON logging for debugging
  - Circuit breaker pattern for graceful degradation
  - Alert management system for critical events
  - Real-time operational dashboard UI

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

- **Monitoring Dashboard** ([monitoring.html](client/monitoring.html))
  - Real-time system health visualization (auto-refresh every 5 seconds)
  - Component status monitoring (Database, YOLO, HTTP client, WebSocket, Cache, Tracker, Feed source)
  - System metrics (CPU, memory, disk usage)
  - Feed statistics with availability progress bars
  - Performance metrics (cache size, active connections, circuit breaker state)
  - Database statistics (detection count, track count)
  - Recent alerts display with severity levels
  - Database reset functionality
  - Dark theme responsive UI

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
                            │                  │
                            │                  │──HTTPS─▶ ChatSurfer (NRO)
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

**Access the monitoring dashboard:** http://localhost:8000/monitoring.html

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

### Health & Monitoring Endpoints

#### GET /health
Comprehensive health check for all system components

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-21T10:30:00Z",
  "components": [
    {
      "component": "database",
      "status": "healthy",
      "message": "Database connection healthy",
      "latency_ms": 1.2
    },
    {
      "component": "yolo_model",
      "status": "healthy",
      "message": "YOLO model loaded and ready"
    }
  ],
  "system": {
    "cpu_percent": 25.3,
    "memory": {
      "total_gb": 16.0,
      "used_gb": 8.5,
      "percent": 53.1
    },
    "disk": {
      "total_gb": 500.0,
      "used_gb": 250.0,
      "percent": 50.0
    }
  }
}
```

**Components monitored:**
- Database connectivity & performance
- YOLO model availability
- HTTP client health
- WebSocket manager status
- Cache health & size
- Vehicle tracker status
- Feed source availability

#### GET /health/live
Kubernetes liveness probe - simple check if server is running

**Response:** `{"status": "alive", "timestamp": "..."}`

#### GET /health/ready
Kubernetes readiness probe - check if system is ready for traffic

**Response:** `{"status": "ready", "timestamp": "..."}` (200 if ready, 503 if not)

#### GET /metrics
Prometheus metrics export endpoint for monitoring and alerting

**Metrics categories:**
- **Detection metrics**: `cctv_detections_total`, `cctv_detection_confidence`, `cctv_tracks_total`
- **Performance metrics**: `cctv_yolo_inference_seconds`, `cctv_feed_fetch_seconds`, `cctv_cycle_duration_seconds`
- **System metrics**: `cctv_feeds_total`, `cctv_feeds_online`, `cctv_cache_size_bytes`, `cctv_active_websockets`
- **Error metrics**: `cctv_errors_total`, `cctv_feed_failures_total`

**Example queries:**
```promql
# Feed availability percentage
(cctv_feeds_online / cctv_feeds_total) * 100

# Detection rate per minute
rate(cctv_detections_total[1m])

# P95 YOLO inference time
histogram_quantile(0.95, rate(cctv_yolo_inference_seconds_bucket[5m]))
```

#### GET /api/operational/status
Operational dashboard data endpoint

**Response:**
```json
{
  "uptime_seconds": 3600,
  "uptime_human": "1h 0m",
  "feeds": {
    "total": 2402,
    "online": 1876,
    "offline": 526,
    "online_percentage": 78.1,
    "with_vehicles": 145
  },
  "cache": {
    "size_mb": 450.5,
    "items": 2402,
    "avg_size_kb": 192.0
  },
  "components": {
    "yolo_model": "healthy",
    "database": "healthy",
    "tracker": "healthy",
    "websocket": "healthy"
  },
  "websocket": {
    "active_connections": 2
  },
  "circuit_breakers": {
    "feed_fetcher": {
      "state": "closed",
      "failure_count": 0
    }
  },
  "alerts": {
    "recent": [
      {
        "timestamp": "2025-11-21T10:30:00Z",
        "type": "feed_source_degraded",
        "severity": "warning",
        "message": "Feed availability below threshold"
      }
    ]
  },
  "database": {
    "detections_count": 125000,
    "tracks_count": 45000
  }
}
```

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

**Request (CoT):**
```json
{
  "enabled": true,
  "format": "cot",
  "ip": "127.0.0.1",
  "port": 8087
}
```

**Request (Lattice):**
```json
{
  "enabled": true,
  "format": "lattice",
  "latticeUrl": "https://your-env.lattice.anduril.com",
  "latticeToken": "your-env-token",
  "latticeSandboxToken": "your-sandbox-token",
  "latticeIntegration": "taiwan-cctv"
}
```

**Request (ChatSurfer):**
```json
{
  "enabled": true,
  "format": "chatsurfer",
  "chatsurferSession": "your-session-cookie",
  "chatsurferRoom": "your-room-name",
  "chatsurferNickname": "CCTV_Bot",
  "chatsurferDomain": "chatsurferxmppunclass"
}
```

#### GET /api/stream/config
Get current Stream Out configuration

### Database Management Endpoints

#### POST /api/database/reset
Reset database - clear all detections and tracks

**Request:** POST with empty body

**Response:**
```json
{
  "status": "success",
  "message": "Database reset successfully",
  "deleted": {
    "detections": 125000,
    "tracks": 45000
  }
}
```

**Note:** This triggers a warning alert in the alert system

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

### ChatSurfer (NRO)
Send vehicle detection alerts to ChatSurfer chat rooms.

**Config:** Format, Session Cookie, Room Name, Nickname, Domain ID

**Message format:**
```
[VEHICLE DETECTION]
Road: 國道1號
Location: 12K+500
Camera: CCTV-42-0020-162-001
Coords: 25.0478, 121.5319
Time: 2026-02-04 15:30:22 UTC
Snapshot: http://192.168.1.100:8001/api/feeds/CCTV-42-0020-162-001/snapshot
```

**Setup:**
1. Get your ChatSurfer SESSION cookie from browser dev tools
2. Identify the room name you want to post to
3. Configure via the Stream Out panel in the UI or POST to `/api/stream/config`

**Note:** The snapshot URL must be accessible from where ChatSurfer users are located

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

## Operational Monitoring

### Health Checks
The system provides comprehensive health monitoring for all components:

- **Database**: Connection health, query latency, connection count
- **YOLO Model**: Model availability and readiness
- **HTTP Client**: Feed fetcher operational status
- **WebSocket Manager**: Connection manager health
- **Cache**: Memory usage and size limits
- **Vehicle Tracker**: Tracking system status
- **Feed Source**: Feed availability and online percentage

### Prometheus Metrics
Export metrics for monitoring with Prometheus/Grafana:

```bash
# Scrape metrics
curl http://localhost:8001/metrics

# Example Prometheus config
scrape_configs:
  - job_name: 'cctv-viewer'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

**Key metrics to monitor:**
- Feed availability: `(cctv_feeds_online / cctv_feeds_total) * 100` → Target: >80%
- Detection latency: `histogram_quantile(0.95, cctv_yolo_inference_seconds_bucket)` → Target: <200ms
- Error rate: `rate(cctv_errors_total[5m])` → Target: <5%
- Cache size: `cctv_cache_size_bytes / 1024^3` → Target: <1GB

### Structured Logging
All logs are output in JSON format for easy parsing:

```bash
# View logs with jq
tail -f logs/cctv.log | jq

# Filter by level
tail -f logs/cctv.log | jq 'select(.level == "ERROR")'

# Watch specific component
tail -f logs/cctv.log | jq 'select(.context.component == "database")'
```

### Circuit Breaker
Protects against cascading failures:
- Monitors feed fetcher operations
- Opens after 10 consecutive failures
- 5-minute recovery timeout before retrying
- States: closed (normal), open (disabled), half_open (testing)

Check state: `curl http://localhost:8001/api/operational/status | jq '.circuit_breakers'`

### Alert Management
Tracks critical system events with 5-minute cooldown:

**Alert types:**
- `startup_failure` - Component failed to start (critical)
- `database_init_failure` - Database initialization error (critical)
- `high_error_rate` - Error rate threshold exceeded (error)
- `feed_source_degraded` - Too many feeds offline (warning)
- `cache_overflow` - Cache size exceeds limits (warning)

View recent alerts: `curl http://localhost:8001/api/operational/status | jq '.alerts.recent'`

### Monitoring Dashboard
Access the visual monitoring dashboard at http://localhost:8000/monitoring.html

Features:
- Real-time component health visualization
- System metrics (CPU, memory, disk)
- Feed statistics with animated progress bars
- Performance metrics
- Database statistics
- Recent alerts display
- Database reset button
- Auto-refresh every 5 seconds

### Quick Health Check Commands

```bash
# Check overall system status
curl http://localhost:8001/health | jq '.status'

# Get unhealthy components
curl http://localhost:8001/health | jq '.components[] | select(.status != "healthy")'

# Check feed availability percentage
curl http://localhost:8001/api/operational/status | jq '.feeds.online_percentage'

# Watch system health (refreshes every 2 seconds)
watch -n 2 'curl -s http://localhost:8001/health | jq ".status, .system"'
```

## Technical Details

- **Feeds**: 2,402 Taiwan highway camera feeds
- **Backend**: Python 3.8+, FastAPI, YOLOv8, SQLAlchemy, aiosqlite
- **Frontend**: Vanilla JavaScript, Leaflet.js for maps
- **Memory**: ~450 MB for cached JPEGs
- **Refresh Rate**: 2 seconds (grid), 0.5 seconds (single feed)
- **Detection Rate**: ~1876 feeds processed per cycle (~20-30 seconds)
- **Observability**: Prometheus metrics, structured logging, health checks, circuit breakers

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
- Check `/api/stats` or `/health` endpoint for system health
- Check feed availability: `curl http://localhost:8001/api/operational/status | jq '.feeds'`
- View monitoring dashboard: http://localhost:8000/monitoring.html

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
- Monitor cache size: `curl http://localhost:8001/api/operational/status | jq '.cache'`
- Check system memory: `curl http://localhost:8001/health | jq '.system.memory'`

### Map markers not updating
- Check browser console for "Map: WebSocket connected"
- Markers update in real-time when vehicles detected
- Try refreshing the page to reconnect WebSocket

## Project Structure

```
videoviewer/
├── backend/
│   ├── main.py                 # FastAPI server + YOLO + WebSocket + monitoring
│   ├── database.py             # SQLAlchemy models and DB manager
│   ├── tracker.py              # Multi-object tracking
│   ├── websocket_manager.py    # WebSocket connection handler
│   └── observability.py        # Health checks, metrics, logging, alerts
├── client/
│   ├── index.html              # Grid view with vehicle detection
│   ├── map.html                # Interactive map view
│   ├── feed.html               # Single feed detailed view
│   └── monitoring.html         # Operational monitoring dashboard
├── data/
│   └── detections.db           # SQLite database (auto-created)
├── logs/
│   └── cctv.log                # Structured JSON logs (auto-created)
├── venv/                       # Python virtual environment
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── start.sh                    # Startup script
├── IMPLEMENTATION_ROADMAP.md   # Feature roadmap
├── OPTIMIZATIONS.md            # Performance optimizations guide
└── README.md                   # This file
```


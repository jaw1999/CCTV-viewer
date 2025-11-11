# Taiwan Highway CCTV Viewer with Real-Time Vehicle Detection

Real-time viewer for 2,500+ Taiwan Highway Bureau CCTV feeds with YOLOv8 vehicle detection and Stream Out integration.

## Features

### Backend
- YOLOv8 vehicle detection (cars, buses, trucks at 60% confidence)
- Stream Out: CoT (UDP) and Lattice (REST API)
- Concurrent ingestion with connection pooling
- In-memory caching
- REST API

### Frontend
- Stream Out configuration UI
- Vehicle detection badges and bounding boxes
- Filter by status or detected vehicles
- Search by location/road/ID
- Auto-refresh thumbnails

## Architecture

```
┌─────────────────┐         ┌──────────────────────────────┐         ┌─────────────────┐
│   Taiwan        │         │   FastAPI Backend            │         │   Web Client    │
│   Highway       │────────▶│   Port 8001                  │────────▶│   (Browser)     │
│   Bureau        │  HTTPS  │   + YOLOv8n Vehicle Detector │   HTTP  │   Port 8000     │
│   (2500 feeds)  │         │   + In-Memory Cache          │         │                 │
└─────────────────┘         └──────────────────────────────┘         └─────────────────┘
                                   │
                                   │ Concurrent Ingestion
                                   │ (Sequential Batches of 200)
                                   │ + YOLO Processing
                                   │
                                   ▼
                            ┌──────────────┐
                            │ Memory Cache │
                            │ (Annotated   │
                            │  Frames)     │
                            └──────────────┘
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

### GET /api/feeds
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

### GET /api/feeds/{feed_id}/snapshot
Returns cached JPEG snapshot for a specific feed (with YOLO bounding boxes if vehicles detected)

### GET /api/feeds/{feed_id}/stream
Returns live MJPEG stream (for future use)

### GET /api/stats
Returns system statistics

**Response:**
```json
{
  "totalFeeds": 2402,
  "cachedFeeds": 2402,
  "workingFeeds": 1876,
  "offlineFeeds": 526,
  "lastUpdate": 1699999999.0,
  "cacheSize": 450.5
}
```

### POST /api/stream/config
Configure Stream Out integration

**Request:**
```json
{
  "enabled": true,
  "format": "cot",
  "ip": "127.0.0.1",
  "port": 8087,
  "latticeToken": "",
  "latticeSandboxToken": "",
  "latticeIntegration": "taiwan-cctv",
  "latticeUrl": ""
}
```

### GET /api/stream/config
Get current Stream Out configuration

## Stream Out Integration

### CoT (Cursor on Target)
Send vehicle detection events to TAK/ATAK systems via UDP.

**Config:** Format, IP, Port

**Message:** XML event with camera location, metadata, timestamp, video link

### Lattice (Anduril)
Publish vehicle track entities to Lattice platform.

**Config:** Format, URL, Environment Token, Sandbox Token (for sandboxes), Integration Name

**Entity:** Camera ID, location, VEHICLE platform type, 1-hour expiry

**Note:** Sandboxes need two tokens (Authorization + anduril-sandbox-authorization headers)

## YOLO Vehicle Detection

- Model: YOLOv8n
- Classes: Car, Bus, Truck (COCO 2, 5, 7)
- Confidence: 60%
- Min box size: 30px
- Processing: ~50-100ms per image

## Technical Details

- 2,402 Taiwan highway camera feeds
- Python 3.8+, FastAPI, YOLOv8
- Vanilla JavaScript frontend
- ~450 MB memory for cached JPEGs
- 1-2 second refresh rate

## Dependencies

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
httpx>=0.25.0
pillow>=10.0.0
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
anduril-lattice-sdk>=1.0.0
```

The YOLOv8n model (~6MB) will be automatically downloaded on first run.

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
- Some feeds may actually be offline (normal)
- Check `/api/stats` endpoint for system health

### YOLO not detecting vehicles
- Check backend logs for "YOLO model loaded successfully"
- Ensure PyTorch and dependencies installed correctly
- Detection requires confidence ≥0.6 (60%) and minimum box size of 30 pixels
- Only detects cars, buses, and trucks (motorcycles excluded)
- Some cameras may not have vehicles in frame

### Slow refresh rate
- Backend logs show actual fetch time + YOLO processing time
- Network latency to Taiwan servers affects speed
- YOLO adds ~50-100ms per image
- Adjust `REFRESH_INTERVAL` in frontend code if needed

### High memory usage
- Each cached JPEG is ~200 KB
- 2,400 feeds = ~480 MB baseline
- This is normal and expected

## Project Structure

```
videoviewer/
├── backend/
│   └── main.py              # FastAPI server + YOLOv8 integration
├── client/
│   └── index.html           # Frontend UI with vehicle detection
├── venv/                    # Python virtual environment
├── requirements.txt         # Python dependencies
├── start.sh                 # Startup script
└── README.md                # This file
```
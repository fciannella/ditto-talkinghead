# Ditto TalkingHead Streaming Services Guide

This guide explains how to use the real-time streaming services for Ditto TalkingHead generation.

## 📋 Overview

Two streaming services are available:

1. **WebSocket Service** (`streaming_service.py`) - Browser-based real-time streaming
2. **RTMP Service** (`rtmp_streaming_service.py`) - Production RTMP streaming for platforms like YouTube, Twitch

## 🎯 Service 1: WebSocket Streaming Service

### Features:
- Real-time browser-based streaming
- WebRTC audio input from microphone
- Live video output in browser
- Perfect for demos and testing

### Usage:

1. **Start the service:**
```bash
cd /app/src
python streaming_service.py "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" "/app/checkpoints/ditto_trt_Ampere_Plus"
```

2. **Open browser:**
```
http://localhost:8000
```

3. **Use the web interface:**
   - Click "Start Streaming" to begin
   - Allow microphone access
   - Speak into microphone
   - See real-time talking head generation

### API Endpoints:
- `GET /` - Demo web page
- `WebSocket /ws` - Real-time streaming endpoint

## 🎯 Service 2: RTMP Streaming Service

### Features:
- Production-ready RTMP streaming
- Stream to YouTube Live, Twitch, Facebook Live
- Real-time microphone audio capture
- RESTful API for control

### Setup:

1. **Start the service:**
```bash
cd /app/src
python rtmp_streaming_service.py "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" "/app/checkpoints/ditto_trt_Ampere_Plus"
```

2. **Get RTMP URL from your platform:**

**YouTube Live:**
- Go to YouTube Studio → Go Live → Stream
- Copy the Stream URL and Stream Key
- Format: `rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY`

**Twitch:**
- Go to Twitch Creator Dashboard → Settings → Stream
- Copy Server and Stream Key  
- Format: `rtmp://live.twitch.tv/app/YOUR_STREAM_KEY`

3. **Start streaming:**
```bash
curl -X POST "http://localhost:8000/start_stream/my_stream" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "/app/data/avatar_image.png",
    "rtmp_url": "rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY"
  }'
```

4. **Monitor stream:**
```bash
# List active streams
curl http://localhost:8000/streams

# Health check
curl http://localhost:8000/health
```

5. **Stop streaming:**
```bash
curl -X DELETE http://localhost:8000/stop_stream/my_stream
```

### API Endpoints:
- `POST /start_stream/{stream_id}` - Start RTMP stream
- `DELETE /stop_stream/{stream_id}` - Stop RTMP stream  
- `GET /streams` - List active streams
- `GET /health` - Health check

## 🔧 Configuration Options

### Performance Tuning:

```python
setup_kwargs = {
    "online_mode": True,
    "sampling_timesteps": 25,    # Lower = faster, higher = better quality
    "crop_scale": 2.3,
    "max_size": 512,            # Lower = faster, higher = better quality
    "crop_vx_ratio": 0,
    "crop_vy_ratio": -0.125,
}
```

### Audio Settings:

```python
# Audio capture settings
sample_rate = 16000       # Required by the model
chunk_size = 1024        # Lower = lower latency, higher = more stable
```

### Video Settings:

```python
# RTMP streaming settings
width = 512              # Output width
height = 512             # Output height  
fps = 25                # Frames per second
```

## 🚀 Production Deployment

### 1. Docker Deployment:

```bash
# Build container with streaming services
docker build -t ditto-streaming .

# Run with GPU support
docker run -it --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/data:/app/data \
  -p 8000:8000 \
  --name ditto-streaming \
  ditto-streaming \
  bash

# Inside container
cd /app/src
python rtmp_streaming_service.py "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" "/app/checkpoints/ditto_trt_Ampere_Plus"
```

### 2. Multiple Streams:

```bash
# Start multiple streams simultaneously
curl -X POST "http://localhost:8000/start_stream/youtube_stream" \
  -H "Content-Type: application/json" \
  -d '{"source_path": "/app/data/avatar1.png", "rtmp_url": "rtmp://youtube.com/live2/KEY1"}'

curl -X POST "http://localhost:8000/start_stream/twitch_stream" \
  -H "Content-Type: application/json" \
  -d '{"source_path": "/app/data/avatar2.png", "rtmp_url": "rtmp://twitch.tv/app/KEY2"}'
```

### 3. Load Balancing:

For high-scale deployment:
- Run multiple service instances on different ports
- Use nginx for load balancing
- Consider Redis for session management

## 🔍 Troubleshooting

### Common Issues:

1. **Audio not working:**
   - Check microphone permissions
   - Verify PyAudio installation: `python -c "import pyaudio"`
   - Check ALSA/audio device access in container

2. **RTMP stream fails:**
   - Verify RTMP URL and stream key
   - Check ffmpeg installation: `ffmpeg -version`
   - Test with simple ffmpeg command first

3. **Performance issues:**
   - Reduce `max_size` (e.g., 256 instead of 512)
   - Increase `sampling_timesteps` for better quality
   - Monitor GPU memory usage

4. **Container audio issues:**
   - Run container with audio device access:
   ```bash
   docker run --device /dev/snd --gpus all ...
   ```

### Debugging:

```bash
# Check audio devices
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(p.get_device_info_by_index(i)) for i in range(p.get_device_count())]"

# Test RTMP manually
ffmpeg -f lavfi -i testsrc=duration=10:size=320x240:rate=30 -f flv rtmp://your_url

# Monitor GPU usage
nvidia-smi -l 1
```

## 📊 Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│   Ditto Pipeline │───▶│  Video Output   │
│  (Microphone)   │    │                  │    │   (RTMP/WS)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Worker Threads │
                    │                  │
                    │ • Audio2Motion   │
                    │ • MotionStitch   │
                    │ • WarpF3D        │
                    │ • DecodeF3D      │
                    │ • PutBack        │
                    │ • Writer         │
                    └──────────────────┘
```

The system maintains real-time performance through:
- **Multi-threading**: Each pipeline stage runs in parallel
- **Queue-based communication**: Non-blocking data flow
- **Optimized settings**: Reduced quality for faster processing
- **Smart buffering**: Audio and video frame queues

## 🎭 Use Cases

1. **Virtual Presentations**: Real-time avatar for video calls
2. **Live Streaming**: YouTube/Twitch with virtual presenter
3. **Interactive Demos**: Trade shows, exhibitions
4. **Content Creation**: Automated video generation
5. **Accessibility**: Voice-driven avatar for communication

This streaming setup transforms the ditto-talkinghead from a batch processing tool into a real-time interactive service! 
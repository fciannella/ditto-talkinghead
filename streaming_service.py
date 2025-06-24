import asyncio
import base64
import io
import json
import threading
import time
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from stream_pipeline_online import StreamSDK


class StreamingVideoWriter:
    """Custom video writer that streams frames via WebSocket instead of saving to file"""
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.frame_queue = asyncio.Queue(maxsize=30)  # Buffer up to 30 frames
        self.is_active = True
        
    def __call__(self, frame_rgb, fmt="rgb"):
        """Called by the pipeline to write frames"""
        if not self.is_active:
            return
            
        # Convert RGB to BGR for OpenCV
        if fmt == "rgb":
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_rgb
            
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Add to queue (non-blocking)
        try:
            self.frame_queue.put_nowait({
                "type": "frame",
                "data": frame_base64,
                "timestamp": time.time()
            })
        except asyncio.QueueFull:
            # Drop oldest frame if queue is full
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait({
                    "type": "frame", 
                    "data": frame_base64,
                    "timestamp": time.time()
                })
            except asyncio.QueueEmpty:
                pass
    
    def close(self):
        """Stop the writer"""
        self.is_active = False
        try:
            self.frame_queue.put_nowait({"type": "end"})
        except asyncio.QueueFull:
            pass


class TalkingHeadStreamingService:
    """Real-time talking head streaming service"""
    
    def __init__(self, cfg_pkl: str, data_root: str):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.active_sessions = {}
        
    def create_session(self, session_id: str, source_path: str, websocket: WebSocket):
        """Create a new streaming session"""
        # Initialize SDK
        sdk = StreamSDK(self.cfg_pkl, self.data_root)
        
        # Custom streaming writer
        streaming_writer = StreamingVideoWriter(websocket)
        
        # Setup SDK with streaming output
        setup_kwargs = {
            "online_mode": True,
            "sampling_timesteps": 25,  # Faster for real-time
            "crop_scale": 2.3,
            "max_size": 512,  # Smaller for faster processing
        }
        
        # Monkey patch the writer to use our streaming writer
        sdk.setup(source_path, "/dev/null", **setup_kwargs)  # dummy output path
        sdk.writer = streaming_writer
        
        self.active_sessions[session_id] = {
            "sdk": sdk,
            "writer": streaming_writer,
            "websocket": websocket,
            "active": True
        }
        
        return sdk
    
    def process_audio_chunk(self, session_id: str, audio_data: np.ndarray):
        """Process audio chunk for a session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if session["active"]:
                session["sdk"].run_chunk(audio_data)
    
    def close_session(self, session_id: str):
        """Close a streaming session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session["active"] = False
            session["sdk"].close()
            session["writer"].close()
            del self.active_sessions[session_id]


# FastAPI service
app = FastAPI(title="Ditto TalkingHead Streaming Service")

# Global service instance
service = None

def init_service(cfg_pkl: str, data_root: str):
    global service
    service = TalkingHeadStreamingService(cfg_pkl, data_root)


@app.get("/")
async def get_demo_page():
    """Serve a demo HTML page"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ditto TalkingHead Streaming Demo</title>
    </head>
    <body>
        <h1>Ditto TalkingHead Real-time Streaming</h1>
        <div id="status" style="margin-bottom: 10px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
            <strong>Status:</strong> <span id="connectionStatus">Disconnected</span><br>
            <strong>WebSocket URL:</strong> <span id="wsUrl">Not set</span>
        </div>
        <div>
            <video id="output" width="512" height="512" autoplay muted></video>
        </div>
        <div>
            <button id="startBtn">Start Streaming</button>
            <button id="stopBtn">Stop Streaming</button>
        </div>
        <div id="errorMsg" style="color: red; margin-top: 10px;"></div>
        
        <script>
            let ws = null;
            let mediaRecorder = null;
            let isStreaming = false;
            
            const video = document.getElementById('output');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const statusEl = document.getElementById('connectionStatus');
            const wsUrlEl = document.getElementById('wsUrl');
            const errorEl = document.getElementById('errorMsg');
            
            document.getElementById('startBtn').onclick = startStreaming;
            document.getElementById('stopBtn').onclick = stopStreaming;
            
            function updateStatus(status, error = '') {
                statusEl.textContent = status;
                errorEl.textContent = error;
                if (error) {
                    errorEl.style.display = 'block';
                } else {
                    errorEl.style.display = 'none';
                }
            }
            
            async function startStreaming() {
                if (isStreaming) return;
                
                try {
                    updateStatus('Requesting microphone access...');
                    
                    // Get microphone access
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    
                    // Setup WebSocket - dynamically use current host
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const host = window.location.host;
                    const wsUrl = `${protocol}//${host}/ws`;
                    
                    wsUrlEl.textContent = wsUrl;
                    updateStatus('Connecting to server...');
                    
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = () => {
                        console.log('WebSocket connected');
                        updateStatus('Connected - Initializing stream...');
                        // Send start command with source image
                        ws.send(JSON.stringify({
                            type: 'start',
                            source_path: '/app/data/source_image.png'  // Configure this
                        }));
                        updateStatus('Streaming active');
                    };
                    
                    ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                        updateStatus('Connection failed', 'Failed to connect to streaming server. Check if the service is running on the server.');
                    };
                    
                    ws.onclose = (event) => {
                        console.log('WebSocket closed:', event);
                        if (isStreaming) {
                            updateStatus('Disconnected', 'Connection to server lost.');
                        } else {
                            updateStatus('Disconnected');
                        }
                    };
                    
                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        if (data.type === 'frame') {
                            displayFrame(data.data);
                        }
                    };
                
                // Setup audio recording
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=pcm'
                });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                        // Send audio data to server
                        event.data.arrayBuffer().then(buffer => {
                            ws.send(buffer);
                        });
                    }
                };
                
                mediaRecorder.start(200); // 200ms chunks
                isStreaming = true;
                
                } catch (error) {
                    console.error('Error starting stream:', error);
                    updateStatus('Error', `Failed to start streaming: ${error.message}`);
                    if (ws) {
                        ws.close();
                    }
                }
            }
            
            function displayFrame(base64Data) {
                const img = new Image();
                img.onload = () => {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    
                    // Convert canvas to video frame
                    canvas.toBlob(blob => {
                        const url = URL.createObjectURL(blob);
                        video.src = url;
                    }, 'image/jpeg');
                };
                img.src = 'data:image/jpeg;base64,' + base64Data;
            }
            
            function stopStreaming() {
                if (!isStreaming) return;
                
                updateStatus('Stopping...');
                
                if (mediaRecorder) {
                    mediaRecorder.stop();
                }
                if (ws) {
                    ws.close();
                }
                isStreaming = false;
                updateStatus('Disconnected');
            }
        </script>
    </body>
    </html>
    """)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming"""
    await websocket.accept()
    session_id = str(time.time())  # Simple session ID
    
    try:
        # Wait for start command
        start_data = await websocket.receive_text()
        start_msg = json.loads(start_data)
        
        if start_msg["type"] == "start":
            source_path = start_msg["source_path"]
            
            # Create streaming session
            sdk = service.create_session(session_id, source_path, websocket)
            
            # Start frame streaming task
            frame_task = asyncio.create_task(stream_frames(session_id, websocket))
            
            # Process incoming audio
            while True:
                try:
                    # Receive audio data
                    audio_data = await websocket.receive_bytes()
                    
                    # Convert WebRTC audio to numpy array (simplified)
                    # In production, you'd need proper audio decoding
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    
                    # Process audio chunk
                    service.process_audio_chunk(session_id, audio_array)
                    
                except WebSocketDisconnect:
                    break
                    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup
        service.close_session(session_id)


async def stream_frames(session_id: str, websocket: WebSocket):
    """Stream generated frames to client"""
    if session_id not in service.active_sessions:
        return
        
    session = service.active_sessions[session_id]
    writer = session["writer"]
    
    try:
        while session["active"]:
            # Get frame from queue
            frame_data = await writer.frame_queue.get()
            
            if frame_data["type"] == "end":
                break
                
            # Send frame to client
            await websocket.send_text(json.dumps(frame_data))
            
    except Exception as e:
        print(f"Frame streaming error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python streaming_service.py <cfg_pkl> <data_root>")
        sys.exit(1)
    
    cfg_pkl = sys.argv[1]
    data_root = sys.argv[2]
    
    # Initialize service
    init_service(cfg_pkl, data_root)
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000) 
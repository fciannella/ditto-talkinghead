import asyncio
import threading
import time
import numpy as np
import cv2
import subprocess
import queue
import pyaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from stream_pipeline_online import StreamSDK


class RTMPStreamWriter:
    """RTMP video writer using ffmpeg pipe"""
    
    def __init__(self, rtmp_url: str, width: int = 512, height: int = 512, fps: int = 25):
        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.is_active = False
        
    def start(self):
        """Start the RTMP stream"""
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # Input from pipe
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', '30',
            '-sc_threshold', '0',
            '-f', 'flv',
            self.rtmp_url
        ]
        
        self.process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.is_active = True
        print(f"Started RTMP stream to {self.rtmp_url}")
        
    def __call__(self, frame_rgb, fmt="rgb"):
        """Write frame to RTMP stream"""
        if not self.is_active or self.process is None:
            return
            
        try:
            # Ensure frame is RGB and correct size
            if fmt != "rgb":
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            
            if frame_rgb.shape[:2] != (self.height, self.width):
                frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))
            
            # Write to ffmpeg pipe
            self.process.stdin.write(frame_rgb.tobytes())
            self.process.stdin.flush()
            
        except (BrokenPipeError, OSError):
            print("RTMP stream disconnected")
            self.is_active = False
    
    def close(self):
        """Stop the RTMP stream"""
        self.is_active = False
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


class AudioCapture:
    """Real-time audio capture using PyAudio"""
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_active = False
        
    def start_capture(self):
        """Start capturing audio from microphone"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            self.is_active = True
            print("Started audio capture")
        except Exception as e:
            print(f"Failed to start audio capture: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback function"""
        if self.is_active:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            try:
                self.audio_queue.put_nowait(audio_data)
            except queue.Full:
                # Drop oldest audio if queue is full
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_data)
                except queue.Empty:
                    pass
        return (None, pyaudio.paContinue)
    
    def get_audio_chunk(self, timeout=1.0):
        """Get next audio chunk"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_capture(self):
        """Stop audio capture"""
        self.is_active = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.audio.terminate()


class TalkingHeadRTMPService:
    """RTMP Streaming service for talking head generation"""
    
    def __init__(self, cfg_pkl: str, data_root: str):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.active_streams = {}
        
    def create_stream(self, stream_id: str, source_path: str, rtmp_url: str):
        """Create a new RTMP stream"""
        # Initialize SDK
        sdk = StreamSDK(self.cfg_pkl, self.data_root)
        
        # RTMP writer
        rtmp_writer = RTMPStreamWriter(rtmp_url)
        rtmp_writer.start()
        
        # Audio capture
        audio_capture = AudioCapture()
        
        # Setup SDK
        setup_kwargs = {
            "online_mode": True,
            "sampling_timesteps": 25,
            "crop_scale": 2.3,
            "max_size": 512,
        }
        
        sdk.setup(source_path, "/dev/null", **setup_kwargs)
        sdk.writer = rtmp_writer  # Replace writer
        
        self.active_streams[stream_id] = {
            "sdk": sdk,
            "rtmp_writer": rtmp_writer,
            "audio_capture": audio_capture,
            "active": True
        }
        
        return sdk, audio_capture
    
    def start_stream_processing(self, stream_id: str):
        """Start processing audio for a stream"""
        if stream_id not in self.active_streams:
            return
            
        stream = self.active_streams[stream_id]
        stream["audio_capture"].start_capture()
        
        # Start audio processing thread
        def audio_processor():
            while stream["active"]:
                audio_chunk = stream["audio_capture"].get_audio_chunk()
                if audio_chunk is not None:
                    # Process audio chunk
                    stream["sdk"].run_chunk(audio_chunk)
                else:
                    time.sleep(0.01)  # Small delay if no audio
        
        stream["audio_thread"] = threading.Thread(target=audio_processor)
        stream["audio_thread"].start()
    
    def stop_stream(self, stream_id: str):
        """Stop a stream"""
        if stream_id in self.active_streams:
            stream = self.active_streams[stream_id]
            stream["active"] = False
            
            # Stop audio capture
            stream["audio_capture"].stop_capture()
            
            # Wait for audio thread
            if "audio_thread" in stream:
                stream["audio_thread"].join(timeout=5)
            
            # Stop SDK and RTMP
            stream["sdk"].close()
            stream["rtmp_writer"].close()
            
            del self.active_streams[stream_id]


# FastAPI application
app = FastAPI(title="Ditto TalkingHead RTMP Streaming Service")

# Global service
service = None

def init_service(cfg_pkl: str, data_root: str):
    global service
    service = TalkingHeadRTMPService(cfg_pkl, data_root)


class StreamRequest(BaseModel):
    source_path: str
    rtmp_url: str


@app.post("/start_stream/{stream_id}")
async def start_stream(stream_id: str, request: StreamRequest):
    """Start a new RTMP stream"""
    if stream_id in service.active_streams:
        raise HTTPException(status_code=400, detail="Stream already exists")
    
    try:
        # Create stream
        sdk, audio_capture = service.create_stream(
            stream_id, request.source_path, request.rtmp_url
        )
        
        # Start processing
        service.start_stream_processing(stream_id)
        
        return {
            "message": f"Stream {stream_id} started successfully",
            "rtmp_url": request.rtmp_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/stop_stream/{stream_id}")
async def stop_stream(stream_id: str):
    """Stop an RTMP stream"""
    if stream_id not in service.active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    try:
        service.stop_stream(stream_id)
        return {"message": f"Stream {stream_id} stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/streams")
async def list_streams():
    """List active streams"""
    return {
        "active_streams": list(service.active_streams.keys()),
        "count": len(service.active_streams)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ditto-talkinghead-rtmp"}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python rtmp_streaming_service.py <cfg_pkl> <data_root>")
        sys.exit(1)
    
    cfg_pkl = sys.argv[1]
    data_root = sys.argv[2]
    
    # Initialize service
    init_service(cfg_pkl, data_root)
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000) 
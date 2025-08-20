import asyncio
import base64
import json
import uuid
import websockets
from typing import Awaitable, Callable, Optional
from asyncio import Queue

from pipecat.frames.frames import AudioRawFrame, Frame, StartFrame, EndFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport, TransportParams


class TwilioInputProcessor(FrameProcessor):
    def __init__(self, transport: "TwilioWebsocketTransport"):
        super().__init__()
        self._transport = transport
        self._frame_queue = Queue()
        self._queue_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Start the WebSocket server when we receive the first StartFrame
        if isinstance(frame, StartFrame):
            await self._transport._start_server()
            # Start processing queued frames
            self._queue_task = asyncio.create_task(self._process_frame_queue())
        elif isinstance(frame, EndFrame):
            await self._transport._stop_server()
            if self._queue_task:
                self._queue_task.cancel()
            
        await self.push_frame(frame, direction)
        
    async def queue_audio_frame(self, audio_frame: AudioRawFrame):
        """Queue audio frame from Twilio to be processed by the pipeline"""
        await self._frame_queue.put(audio_frame)
        
    async def _process_frame_queue(self):
        """Process queued audio frames and inject them into the pipeline"""
        try:
            while True:
                audio_frame = await self._frame_queue.get()
                # Reduce spam - only log every 200 frames
                if not hasattr(self, '_inject_count'):
                    self._inject_count = 0
                self._inject_count += 1
                if self._inject_count % 200 == 0:
                    print(f"Injected {self._inject_count} audio frames into pipeline")
                await self.push_frame(audio_frame, FrameDirection.DOWNSTREAM)
        except asyncio.CancelledError:
            print("Frame queue processor cancelled")
        except Exception as e:
            print(f"Error processing frame queue: {e}")


class TwilioOutputProcessor(FrameProcessor):
    def __init__(self, transport: "TwilioWebsocketTransport"):
        super().__init__()
        self._transport = transport
        self._bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            # Only log every few frames to avoid spam
            if not hasattr(self, '_audio_out_count'):
                self._audio_out_count = 0
            self._audio_out_count += 1
            if self._audio_out_count % 10 == 0:
                print(f"TTS audio frame #{self._audio_out_count}: {len(frame.audio)} bytes")
            if not self._bot_speaking:
                self._bot_speaking = True
                # Start one reset task when bot starts speaking
                asyncio.create_task(self._reset_speaking_flag())
            await self._transport.send_audio(frame)
            
        await self.push_frame(frame, direction)
        
    async def _reset_speaking_flag(self):
        """Reset speaking flag after bot finishes speaking"""
        await asyncio.sleep(1.0)  # Wait 1 second after audio frame
        self._bot_speaking = False
        print("Bot finished speaking - listening resumed")
        
    def is_bot_speaking(self):
        """Check if bot is currently speaking"""
        return self._bot_speaking


class TwilioWebsocketTransport(BaseTransport):
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        super().__init__()
        self._host = host
        self._port = port
        self._websocket = None
        self._server = None
        self._server_task = None
        self._stream_sid = None
        self._input_processor = TwilioInputProcessor(self)
        self._output_processor = TwilioOutputProcessor(self)

    def input(self) -> FrameProcessor:
        return self._input_processor

    def output(self) -> FrameProcessor:
        return self._output_processor

    async def _start_server(self):
        if not self._server:
            print(f"Starting Twilio WebSocket server on {self._host}:{self._port}")
            
            # Create wrapper to handle websockets.serve signature
            async def websocket_handler(websocket, path=None):
                print(f"Incoming WebSocket connection from: {websocket.remote_address}")
                print(f"WebSocket path: {path}")
                await self._handle_websocket(websocket, path)
            
            # Start WebSocket server
            self._server = await websockets.serve(
                websocket_handler,
                self._host,
                self._port
            )
            print(f"WebSocket server started on {self._host}:{self._port}")
            print(f"Server listening for connections...")

    async def _stop_server(self):
        print("Stopping Twilio WebSocket server")
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

    async def _handle_websocket(self, websocket, path=None):
        """Handle incoming WebSocket connection from Twilio"""
        print("Twilio WebSocket connection established")
        self._websocket = websocket
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_twilio_message(data)
                    
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                except Exception as e:
                    print(f"Error processing Twilio message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("Twilio WebSocket connection closed")
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self._websocket = None
            print("Twilio WebSocket handler ended")

    async def _process_twilio_message(self, data: dict):
        """Process incoming message from Twilio"""
        event = data.get("event")
        
        if event == "connected":
            print(f"Twilio connected: {data}")
            
        elif event == "start":
            print(f"Twilio stream started: {data}")
            # Capture stream ID for sending audio back
            start_data = data.get("start", {})
            self._stream_sid = start_data.get("streamSid") or data.get("streamSid")
            print(f"Captured streamSid: {self._stream_sid}")
            
        elif event == "media":
            # Process incoming audio
            media = data.get("media", {})
            payload = media.get("payload", "")
            
            if payload:
                try:
                    # Decode audio from Twilio (μ-law format)
                    audio_data = base64.b64decode(payload)
                    
                    # Debug: log every 50th chunk to avoid spam
                    if not hasattr(self, '_audio_count'):
                        self._audio_count = 0
                    self._audio_count += 1
                    if self._audio_count % 50 == 0:
                        print(f"Audio chunk #{self._audio_count}: {len(audio_data)} bytes")
                    
                    # Convert μ-law to linear PCM
                    import audioop
                    linear_audio = audioop.ulaw2lin(audio_data, 2)
                    
                    # Resample from 8kHz to 16kHz for better STT recognition
                    try:
                        resampled_audio, _ = audioop.ratecv(
                            linear_audio, 2, 1, 8000, 16000, None
                        )
                        sample_rate = 16000
                        final_audio = resampled_audio
                    except Exception as resample_error:
                        print(f"Resampling failed: {resample_error}, using original")
                        final_audio = linear_audio
                        sample_rate = 8000
                    
                    # Create audio frame for pipeline
                    audio_frame = AudioRawFrame(
                        audio=final_audio,
                        sample_rate=sample_rate,
                        num_channels=1
                    )
                    audio_frame.id = str(uuid.uuid4())
                    # Queue audio frame for pipeline processing
                    await self._input_processor.queue_audio_frame(audio_frame)
                    
                except Exception as e:
                    print(f"Error processing audio: {e}")
            # Skip empty payload logging
                    
        elif event == "stop":
            print(f"Twilio stream stopped: {data}")
            await self._input_processor.queue_audio_frame(EndFrame())
            
        else:
            print(f"Unknown Twilio event: {event}")

    async def send_audio(self, frame: AudioRawFrame):
        """Send audio frame back to Twilio"""
        if not self._websocket:
            return
            
        try:
            # Convert linear PCM to μ-law for Twilio
            import audioop
            
            # Ensure audio is 16-bit signed integers
            audio_data = frame.audio
            
            # Resample to 8kHz if needed (Twilio expects 8kHz)
            if frame.sample_rate != 8000:
                try:
                    audio_data, _ = audioop.ratecv(
                        audio_data, 2, frame.num_channels, 
                        frame.sample_rate, 8000, None
                    )
                except Exception as e:
                    print(f"Resampling error: {e}")
                    return
            
            # Split audio into 20ms chunks (160 bytes for 8kHz mono μ-law)
            chunk_size = 320  # 20ms of 8kHz 16-bit mono audio
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad with silence if needed
                    chunk += b'\x00' * (chunk_size - len(chunk))
                
                # Convert to μ-law
                try:
                    mulaw_chunk = audioop.lin2ulaw(chunk, 2)
                except Exception as e:
                    print(f"μ-law conversion error: {e}")
                    continue
                
                # Encode as base64
                payload = base64.b64encode(mulaw_chunk).decode('utf-8')
                
                # Create Twilio media message
                media_message = {
                    "event": "media",
                    "streamSid": self._stream_sid or "",
                    "media": {
                        "payload": payload
                    }
                }
                
                # Send to Twilio
                await self._websocket.send(json.dumps(media_message))
            
            # print(f"Sent {len(audio_data)//chunk_size} audio chunks to Twilio")
            
        except Exception as e:
            print(f"Error sending audio to Twilio: {e}")
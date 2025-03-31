import asyncio
import json
import websockets
from typing import Optional, Any, Dict, List

class WebSocketManager:
    def __init__(self, uri: str = "ws://localhost:8765", max_retries: int = 3, retry_delay: int = 2):
        self.uri = uri
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_message_size = 500000  # ~500KB for safe margin below WebSocket limit

    async def _try_connect(self) -> Optional[websockets.WebSocketClientProtocol]:
        """Attempt to establish a WebSocket connection with retries."""
        for attempt in range(self.max_retries):
            try:
                print(f"Attempting to connect to TTS server at {self.uri}...")
                # Connect with no timeout - wait indefinitely
                return await websockets.connect(self.uri, ping_timeout=None, ping_interval=None, close_timeout=None, max_size=10 * 1024 * 1024)
            except (websockets.exceptions.WebSocketException, ConnectionRefusedError) as e:
                if attempt == self.max_retries - 1:
                    print(f"ERROR: Could not connect to TTS server at {self.uri}. Please make sure the server is running.")
                    print(f"To start the TTS server, run the following commands:")
                    print(f"  cd TTS-Provider")
                    print(f"  python run_server.py")
                    raise ConnectionError(f"Failed to connect to TTS service after {self.max_retries} attempts: {e}")
                print(f"Connection attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)
        return None

    def _split_text(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into smaller chunks that won't exceed the WebSocket frame size limit."""
        # If text is short enough, return it as a single chunk
        if len(text) <= max_length:
            return [text]
            
        chunks = []
        # Try to split on sentence boundaries
        sentences = text.replace('. ', '.\n').replace('! ', '!\n').replace('? ', '?\n').split('\n')
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed the limit, start a new chunk
            if len(current_chunk) + len(sentence) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                # If the sentence itself is too long, split it by words
                if len(sentence) > max_length:
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 > max_length:
                            chunks.append(current_chunk)
                            current_chunk = word
                        else:
                            if current_chunk:
                                current_chunk += " " + word
                            else:
                                current_chunk = word
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    async def send_tts_request(self, text: str, speaker: int, sample_rate: int = 24000, 
                           response_mode: str = "stream", max_audio_length_ms: int = 300000) -> Dict[str, Any]:
        """Send a TTS request and return the response metadata and audio data.
        
        Args:
            text: The text to convert to speech
            speaker: The speaker ID (0 for male, 1 for female)
            sample_rate: The sample rate for the generated audio (default: 24000)
            response_mode: The response mode, either "stream" or "file" (default: "stream")
            max_audio_length_ms: Maximum audio length in milliseconds (default: 300000 - 5 minutes)
            
        Returns:
            Dict containing metadata and audio data
        """
        websocket = await self._try_connect()
        if not websocket:
            raise ConnectionError("Failed to establish connection to TTS service")

        try:
            # Check if the text is too long and needs to be split
            text_length = len(text.encode('utf-8'))
            if text_length > self.max_message_size:
                print(f"Text is too long ({text_length} bytes), splitting into chunks...")
                return await self._process_long_text(websocket, text, speaker, sample_rate, response_mode, max_audio_length_ms)
            
            # Prepare request data using the correct parameter names for TTS-Provider
            request = {
                "text": text,
                "speaker": speaker,  # Changed from voice_id to speaker
                "sample_rate": sample_rate,
                "response_mode": response_mode,  # Add response_mode parameter
                "max_audio_length_ms": max_audio_length_ms  # Add max_audio_length_ms parameter
            }
            
            # Send request
            print(f"Sending TTS request: {json.dumps(request)}")
            await websocket.send(json.dumps(request))
            
            # Get initial response (metadata)
            metadata_str = await websocket.recv()
            response = json.loads(metadata_str)
            
            # Handle loading state or queue state
            status = response.get("status")
            while status in ["loading", "queued"]:
                if status == "loading":
                    print("TTS model is still loading, waiting...")
                elif status == "queued":
                    queue_position = response.get("queue_position", "unknown")
                    print(f"Request queued (position: {queue_position}), waiting...")
                
                await asyncio.sleep(1)
                metadata_str = await websocket.recv()
                response = json.loads(metadata_str)
                status = response.get("status")
            
            if response.get("status") == "success":
                result = {
                    "metadata": response
                }
                
                if response_mode == "file":
                    # In file mode, the server sends only metadata with filepath
                    # We need to copy that file to our local destination
                    result["filepath"] = response.get("filepath")
                else:
                    # In stream mode, the server sends audio data after the metadata
                    # Check expected length vs received length
                    expected_length = response.get("length_bytes", 0)
                    print(f"Expecting to receive {expected_length} bytes of audio data")
                    
                    audio_data = await websocket.recv()
                    actual_length = len(audio_data)
                    
                    print(f"Actually received {actual_length} bytes of audio data")
                    if actual_length < expected_length:
                        print(f"WARNING: Received fewer bytes than expected ({actual_length} < {expected_length})!")
                        print(f"Missing {expected_length - actual_length} bytes ({((expected_length - actual_length) / expected_length) * 100:.1f}% of data)")
                    
                    result["audio_data"] = audio_data
                
                return result
            else:
                error_msg = response.get("message", "Unknown error")
                print(f"TTS service error: {error_msg}")
                raise Exception(f"TTS service error: {error_msg}")
        
        finally:
            # Close the connection when done
            await websocket.close()

    async def _process_long_text(self, websocket, text: str, speaker: int, sample_rate: int, response_mode: str, max_audio_length_ms: int) -> Dict[str, Any]:
        """Process long text by splitting it into chunks, sending separate requests, and combining results."""
        chunks = self._split_text(text)
        print(f"Split text into {len(chunks)} chunks")
        
        all_audio_data = bytearray()
        success_count = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Prepare request for this chunk
                request = {
                    "text": chunk,
                    "speaker": speaker,
                    "sample_rate": sample_rate,
                    "response_mode": "stream",  # Always use stream mode for chunks
                    "max_audio_length_ms": max_audio_length_ms  # Add max_audio_length_ms parameter
                }
                
                # Send request
                print(f"Sending chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                await websocket.send(json.dumps(request))
                
                # Get metadata response
                metadata_str = await websocket.recv()
                response = json.loads(metadata_str)
                
                # Handle loading state or queue state
                status = response.get("status")
                while status in ["loading", "queued"]:
                    if status == "loading":
                        print("TTS model is still loading, waiting...")
                    elif status == "queued":
                        queue_position = response.get("queue_position", "unknown")
                        print(f"Request queued (position: {queue_position}), waiting...")
                    
                    await asyncio.sleep(1)
                    metadata_str = await websocket.recv()
                    response = json.loads(metadata_str)
                    status = response.get("status")
                
                if response.get("status") == "success":
                    # Get the audio data for this chunk
                    chunk_audio = await websocket.recv()
                    all_audio_data.extend(chunk_audio)
                    success_count += 1
                else:
                    error_msg = response.get("message", "Unknown error")
                    print(f"Error processing chunk {i+1}: {error_msg}")
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
        
        if success_count == 0:
            raise Exception("Failed to process any text chunks")
        
        # Create result with combined audio data
        result = {
            "metadata": {
                "status": "success",
                "response_mode": "stream",
                "length_bytes": len(all_audio_data),
                "sample_rate": sample_rate,
                "format": "wav",
                "combined_chunks": len(chunks)
            },
            "audio_data": bytes(all_audio_data)
        }
        
        print(f"Successfully combined audio from {success_count}/{len(chunks)} chunks")
        return result
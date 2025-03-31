import json
import os
from typing import List, Dict
import asyncio
from pydub import AudioSegment

from utils.text_utils import split_text_into_chunks
from utils.websocket_manager import WebSocketManager
from audio.audio_clip import AudioClip

# Update voice mapping to use Sesame's integer-based speaker IDs
VOICES = {
    "Narrator": 0,       # Default male voice
    "AI Debater 1": 1,   # Different voice
    "AI Debater 2": 0,   # Back to the first voice
    "Jane": 1,           # Female voice
    "Valentino": 0       # Male voice
}

def get_segment_audio_file(segment_index):
    """Get the audio file for a specific segment index."""
    clip = AudioClip.from_segment_index(segment_index)
    return clip.file_path if clip else None

def get_segment_duration(audio_file):
    """Get the duration of an audio file in seconds."""
    if audio_file and os.path.exists(audio_file):
        clip = AudioClip(audio_file)
        return clip.duration
    return 5.0  # Default duration if we can't determine

def get_all_timing_data():
    """Get all timing data across all segments for a comprehensive view."""
    all_timing = []
    
    try:
        # Get all timing JSON files
        timing_files = [f for f in os.listdir('outputs/audio_output') if f.endswith('_timing.json')]
        timing_files.sort()  # Ensure proper order
        
        # Absolute start time tracker to adjust all segments to a global timeline
        absolute_start_time = 0.0
        
        for timing_file in timing_files:
            try:
                # Extract the audio file name for this timing file
                audio_file = timing_file.replace('_timing.json', '.wav')
                audio_path = os.path.join('outputs/audio_output', audio_file)
                
                # Get audio duration using AudioClip
                audio_duration = 0.0
                if os.path.exists(audio_path):
                    clip = AudioClip(audio_path)
                    audio_duration = clip.duration
                
                # Load timing data
                with open(os.path.join('outputs/audio_output', timing_file), 'r') as f:
                    timing_data = json.load(f)
                
                # Get the first segment to adjust everything
                if timing_data.get("segments") and len(timing_data["segments"]) > 0:
                    first_segment = timing_data["segments"][0]
                    segment_start = first_segment.get("start_time", 0)
                    
                    # Adjust absolute timing for this file
                    file_offset = absolute_start_time - segment_start
                
                # Process segments and adjust to absolute timeline
                for segment in timing_data.get("segments", []):
                    # Make a copy with adjusted timing
                    adjusted_segment = segment.copy()
                    
                    # Adjust timing to global timeline
                    if "start_time" in adjusted_segment:
                        adjusted_segment["start_time"] += file_offset
                    if "end_time" in adjusted_segment:
                        adjusted_segment["end_time"] += file_offset
                    
                    # Add the audio file reference
                    adjusted_segment["audio_file"] = audio_file
                    
                    # Include segment index for easy reference
                    segment_index = timing_file.split("_")[1].split(".")[0]
                    if segment_index.isdigit():
                        adjusted_segment["segment_index"] = int(segment_index)
                    
                    all_timing.append(adjusted_segment)
                
                # Update absolute start time for the next file
                absolute_start_time += audio_duration
                
            except Exception as e:
                print(f"Error processing timing file {timing_file}: {str(e)}")
                
        print(f"Loaded {len(all_timing)} timing segments across {len(timing_files)} audio files")
        return all_timing
    except Exception as e:
        print(f"Error getting timing data: {str(e)}")
        return []

def parse_debate():
    """Parse the debate.txt file to map text segments to speakers."""
    segments = []
    try:
        with open('outputs/debate.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_speaker = None
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line defines a speaker
            if ":" in line and line.split(":")[0] in ["Narrator", "AI Debater 1", "AI Debater 2", "Jane", "Valentino", "Ground Statement", "Summary", "Result"]:
                # If we have a previous segment, add it
                if current_speaker and current_text:
                    segments.append({
                        "speaker": current_speaker,
                        "text": current_text
                    })
                
                parts = line.split(":", 1)
                current_speaker = parts[0]
                current_text = parts[1].strip()
            else:
                # Continue previous segment
                if current_text:
                    current_text += " " + line
                else:
                    current_text = line
        
        # Add the last segment
        if current_speaker and current_text:
            segments.append({
                "speaker": current_speaker,
                "text": current_text
            })
                    
        return segments
    except Exception as e:
        print(f"Error parsing debate: {str(e)}")
        return []

def get_segment_timing(audio_file):
    """Get the timing information for a specific audio segment."""
    # First, try to get all comprehensive timing data
    all_timing = getattr(get_segment_timing, 'all_timing', None)
    if all_timing is None:
        all_timing = get_all_timing_data()
        
        # Cache it for future calls
        get_segment_timing.all_timing = all_timing
        
    # If no audio file specified, return all timing
    if audio_file is None:
        return all_timing
        
    # If we need specific segment timing, use AudioClip to get the file's timing
    try:
        if os.path.exists(audio_file):
            clip = AudioClip(audio_file)
            return clip.timing_data.get("segments", [])
        else:
            # Return the relevant portion of the global timing data
            return all_timing
    except Exception as e:
        print(f"Warning: Could not read timing file for {audio_file}: {str(e)}")
        # Fall back to global timing data
        return all_timing

def validate_audio_timing(output_file, timing_data):
    """Validates that the audio and timing data are properly aligned.
    
    Args:
        output_file: Path to the audio file
        timing_data: The timing data dictionary with segments
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_wav(output_file)
        audio_duration = len(audio) / 1000.0  # Convert ms to seconds
        
        # Extract timing info
        segments = timing_data.get("segments", [])
        if not segments:
            print(f"WARNING: No segments found in timing data for {output_file}")
            return False
        
        # Calculate total timing coverage
        last_segment = segments[-1]
        timing_duration = last_segment.get("end_time", 0)
        
        # Validate end time matches audio duration
        if abs(timing_duration - audio_duration) > 0.5:  # Allow 0.5s difference
            print(f"WARNING: Timing duration ({timing_duration:.2f}s) doesn't match audio duration ({audio_duration:.2f}s)")
            
            # Fix timing if needed
            for segment in segments:
                # Scale all timings to match audio duration
                segment["start_time"] = (segment["start_time"] / timing_duration) * audio_duration
                segment["end_time"] = (segment["end_time"] / timing_duration) * audio_duration
            
            # Ensure the last segment ends exactly at the audio duration
            segments[-1]["end_time"] = audio_duration
            print(f"Adjusted timing to match audio duration: {audio_duration:.2f}s")
        
        # Check for gaps or overlaps
        for i in range(1, len(segments)):
            prev_end = segments[i-1].get("end_time", 0)
            curr_start = segments[i].get("start_time", 0)
            
            # Check for gaps
            if curr_start - prev_end > 0.2:  # Gap greater than 200ms
                print(f"WARNING: Gap between segments {i-1} and {i}: {curr_start - prev_end:.2f}s")
                # Fix the gap
                segments[i-1]["end_time"] = curr_start
            
            # Check for overlaps
            if prev_end > curr_start:
                print(f"WARNING: Overlap between segments {i-1} and {i}: {prev_end - curr_start:.2f}s")
                # Fix the overlap
                segments[i-1]["end_time"] = curr_start
        
        # Log segment info for debugging
        total_text_length = sum(len(segment.get("text", "")) for segment in segments)
        print(f"Audio: {audio_duration:.2f}s, {len(segments)} segments, ~{total_text_length} chars")
        for i, segment in enumerate(segments):
            segment_duration = segment.get("end_time", 0) - segment.get("start_time", 0)
            chars_per_second = len(segment.get("text", "")) / segment_duration if segment_duration > 0 else 0
            print(f"  Segment {i}: {segment_duration:.2f}s, {len(segment.get('text', ''))} chars ({chars_per_second:.1f} chars/sec)")
            if chars_per_second > 30:  # Very fast speech
                print(f"    WARNING: Segment {i} may be too fast to read")
        
        return True
    except Exception as e:
        print(f"Error validating timing: {e}")
        return False

async def text_to_speech(text: str, voice: int, output_file: str, max_retries: int = 3) -> bool:
    """Convert text to speech using WebSocket TTS service."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Set a reasonable max_audio_length_ms value to handle long texts without model issues
        # 150 seconds should be enough for most debate segments without stressing the model
        max_audio_length_ms = 150000  # 2.5 minutes
        
        # Try multiple times in case of connection issues
        for retry in range(max_retries):
            try:
                # Create WebSocket manager without timeout
                ws_manager = WebSocketManager()  # No timeout, wait indefinitely
                
                print(f"TTS attempt {retry+1}/{max_retries} for {output_file}")
                
                # Send TTS request with correct parameters including higher max_audio_length_ms
                result = await ws_manager.send_tts_request(
                    text=text, 
                    speaker=voice, 
                    sample_rate=24000,
                    response_mode="stream",
                    max_audio_length_ms=max_audio_length_ms
                )
                
                # Extract metadata
                metadata = result.get("metadata", {})
                print(f"TTS response metadata: {json.dumps(metadata)}")
                
                # Add detailed logging about the received audio data
                audio_data = result.get("audio_data", b"")
                print(f"Received audio data size: {len(audio_data)} bytes")
                if len(audio_data) < 100000:  # If suspiciously small
                    print(f"WARNING: Audio data seems unusually small for the text length ({len(text)} chars)")
                
                # Try to save the audio data with file permission handling
                file_saved = False
                for file_retry in range(3):  # Try 3 times to save the file
                    try:
                        # Check if file exists and try to remove it first
                        if os.path.exists(output_file):
                            try:
                                os.remove(output_file)
                                print(f"Removed existing file: {output_file}")
                            except (PermissionError, OSError) as e:
                                print(f"Warning: Could not remove existing file {output_file}: {e}")
                                # Try with a different filename if we can't remove the existing one
                                output_file = output_file.replace('.wav', f'_new_{file_retry}.wav')
                                print(f"Trying alternate filename: {output_file}")
                        
                        # Save the audio data
                        with open(output_file, "wb") as f:
                            f.write(result["audio_data"])
                        
                        file_saved = True
                        break  # File saved successfully
                    except (PermissionError, OSError) as e:
                        print(f"Error saving file on attempt {file_retry+1}: {e}")
                        if file_retry < 2:  # Last retry
                            print(f"Waiting 1 second before retrying...")
                            await asyncio.sleep(1)
                
                if not file_saved:
                    print(f"Failed to save audio file after multiple attempts. Continuing without saving.")
                    return False
                
                # Create timing data based on text chunks
                try:
                    audio = AudioSegment.from_wav(output_file)
                    duration = len(audio) / 1000.0  # Convert ms to seconds
                    
                    # First try to use forced alignment if available (most accurate)
                    try:
                        import torch
                        from gentle_force import ForceAligner
                        
                        print("Attempting to use forced alignment for precise timing...")
                        # Convert audio to the format needed by the aligner
                        temp_audio_path = output_file.replace('.wav', '_temp.wav')
                        audio.export(temp_audio_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                        
                        # Initialize the forced aligner
                        aligner = ForceAligner()
                        
                        # Perform alignment
                        alignment = aligner.align(audio_path=temp_audio_path, text=text)
                        
                        # Process alignment results
                        if alignment and alignment.words:
                            # Group words into reasonable chunks for subtitles
                            timing_data = {"segments": []}
                            current_segment = {"text": "", "start_time": 0, "end_time": 0}
                            word_count = 0
                            
                            for word in alignment.words:
                                # Start a new segment every ~10-15 words or at punctuation
                                if (word_count >= 12 and word.word.endswith(('.', ',', '!', '?', ':', ';'))) or word_count >= 15:
                                    if current_segment["text"]:
                                        current_segment["end_time"] = word.end
                                        timing_data["segments"].append(current_segment)
                                        current_segment = {"text": "", "start_time": word.end, "end_time": 0}
                                        word_count = 0
                                
                                # Add word to current segment
                                if current_segment["text"]:
                                    current_segment["text"] += " " + word.word
                                else:
                                    current_segment["text"] = word.word
                                    current_segment["start_time"] = word.start
                                
                                word_count += 1
                            
                            # Add the last segment if it has content
                            if current_segment["text"]:
                                current_segment["end_time"] = duration
                                timing_data["segments"].append(current_segment)
                            
                            print(f"Successfully created {len(timing_data['segments'])} timing segments using forced alignment")
                            
                            # Clean up temporary file
                            try:
                                os.remove(temp_audio_path)
                            except:
                                pass
                            
                            # Skip the fallback timing method
                            raise StopIteration("Used forced alignment")
                    except ImportError:
                        print("Forced alignment library not available, using fallback timing estimation")
                    except Exception as e:
                        if isinstance(e, StopIteration):
                            raise
                        print(f"Error during forced alignment: {e}, using fallback timing estimation")
                    
                    # Fallback: Improved timing estimation based on natural language features
                    print(f"Creating improved timing data for audio with duration {duration:.2f} seconds")
                    
                    # We'll use natural language features to estimate timing more accurately
                    sentences = []
                    current_sentence = ""
                    # Split text into sentences based on punctuation
                    for char in text:
                        current_sentence += char
                        if char in ['.', '!', '?', ':', ';'] and current_sentence.strip():
                            sentences.append(current_sentence.strip())
                            current_sentence = ""
                    # Add any remaining text as the last sentence
                    if current_sentence.strip():
                        sentences.append(current_sentence.strip())
                    
                    # Calculate character count for each sentence
                    total_chars = sum(len(s) for s in sentences)
                    
                    # Create timing data with a more natural distribution
                    timing_data = {"segments": []}
                    current_time = 0.0
                    
                    for sentence in sentences:
                        # Clean the sentence for better subtitle display
                        sentence = clean_sentence(sentence)
                        if not sentence:  # Skip empty sentences
                            continue
                        
                        # Estimate duration based on sentence length, with adjustments
                        # 1. Base timing: characters in sentence / total characters * total duration
                        # 2. Adjustment for natural pauses between sentences
                        # 3. Adjustment for slower speaking at the beginning and end
                        
                        # Base estimate (proportional to character count)
                        char_ratio = len(sentence) / total_chars
                        sentence_duration = char_ratio * duration
                        
                        # Small adjustment for natural pauses at punctuation (add a little extra time)
                        if sentence.endswith(('.', '!', '?')):
                            sentence_duration += 0.2  # Add 200ms for major punctuation
                        elif sentence.endswith((':', ';')):
                            sentence_duration += 0.1  # Add 100ms for minor punctuation
                        
                        # Create segment
                        segment = {
                            "text": sentence,
                            "start_time": current_time,
                            "end_time": current_time + sentence_duration
                        }
                        timing_data["segments"].append(segment)
                        
                        # Update current time for next segment
                        current_time += sentence_duration
                    
                    # Normalize timing to match actual audio duration
                    if timing_data["segments"] and timing_data["segments"][-1]["end_time"] != duration:
                        # Scale all timings to match the actual audio duration
                        scale_factor = duration / timing_data["segments"][-1]["end_time"]
                        
                        # Apply scaling to all segments
                        for segment in timing_data["segments"]:
                            segment["start_time"] *= scale_factor
                            segment["end_time"] *= scale_factor
                        
                        # Ensure the last segment ends at the exact audio duration
                        if timing_data["segments"]:
                            timing_data["segments"][-1]["end_time"] = duration
                except Exception as e:
                    print(f"Warning: Could not create detailed timing: {str(e)}")
                    # Create at least one basic timing segment with estimated duration
                    estimated_duration = len(text.split()) * 0.3  # Rough estimate: 0.3 seconds per word
                    timing_data = {"segments": [{
                        "text": text,
                        "start_time": 0,
                        "end_time": estimated_duration
                    }]}
                
                # Save timing data with retry
                timing_file = output_file.replace('.wav', '_timing.json')
                for timing_retry in range(3):
                    try:
                        # Validate and potentially correct the timing data
                        validate_audio_timing(output_file, timing_data)
                        
                        # Save the (potentially corrected) timing data
                        with open(timing_file, 'w') as f:
                            json.dump(timing_data, f, indent=2)
                        break
                    except (PermissionError, OSError) as e:
                        print(f"Error saving timing file on attempt {timing_retry+1}: {e}")
                        if timing_retry < 2:
                            await asyncio.sleep(1)
                
                print(f"Successfully generated audio for '{text[:30]}...' (truncated)")
                return True
                
            except ConnectionError as ce:
                print(f"Connection error on attempt {retry+1}: {ce}")
                # Wait before retrying
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2  # Progressive backoff
                    print(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
            except Exception as e:
                print(f"Error in TTS request on attempt {retry+1}: {e}")
                # For any errors, try again if retries remain
                if retry < max_retries - 1:
                    await asyncio.sleep(2)
        
        # If we got here, all retries failed
        print(f"Failed to generate any audio. Creating silent placeholder.")
        
        # Create an empty placeholder file with error handling
        silent_created = False
        for create_retry in range(3):
            try:
                # Try a different filename if needed
                retry_output_file = output_file
                if create_retry > 0:
                    retry_output_file = output_file.replace('.wav', f'_silent_{create_retry}.wav')
                
                # Remove existing file if it exists
                if os.path.exists(retry_output_file):
                    try:
                        os.remove(retry_output_file)
                    except:
                        pass
                
                # Create a silent audio file
                silent_wav = AudioSegment.silent(duration=5000)
                silent_wav.export(retry_output_file, format="wav")
                output_file = retry_output_file  # Update the output file name
                silent_created = True
                break
            except Exception as e:
                print(f"Error creating silent placeholder (attempt {create_retry+1}): {e}")
                if create_retry < 2:
                    await asyncio.sleep(1)
        
        # Create a basic timing file
        if silent_created:
            timing_file = output_file.replace('.wav', '_timing.json')
            try:
                with open(timing_file, 'w') as f:
                    json.dump({
                        "segments": [{
                            "text": text,
                            "start_time": 0,
                            "end_time": 5.0
                        }]
                    }, f, indent=2)
            except Exception as e:
                print(f"Failed to create timing file for silent placeholder: {e}")
        
        return False
            
    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def get_current_subtitle(timing_segments, current_time, default_text=''):
    """Get the current subtitle text based on timing information."""
    # If timing_segments contains proper AudioClip object, use its method
    if isinstance(timing_segments, AudioClip):
        return timing_segments.get_current_subtitle(current_time, default_text)
    
    # Otherwise, use the original implementation
    if not timing_segments or timing_segments is None:
        return default_text, None
    
    # Add a small lookahead buffer (0.1 seconds) to account for processing delay
    buffered_time = current_time + 0.1
    
    # Find the segment that matches the current time
    current_segment = None
    for segment in timing_segments:
        start_time = segment.get('start_time', 0)
        end_time = segment.get('end_time', 0)
        
        # Check if the current time falls within this segment's time range
        if start_time <= buffered_time <= end_time:
            current_segment = segment
            break
        
        # If we've passed the current time and haven't found a match,
        # use the previous segment to avoid showing text too early
        if start_time > buffered_time:
            break
        
        # Keep track of the last valid segment we've seen
        current_segment = segment
    
    if current_segment:
        text = current_segment.get('text', default_text)
        
        # If we're near the end of a segment (within 0.1s), don't show the next one yet
        if current_segment.get('end_time', 0) - current_time < 0.1:
            # Find if there's a gap before the next segment starts
            segment_index = timing_segments.index(current_segment)
            if segment_index + 1 < len(timing_segments):
                next_segment = timing_segments[segment_index + 1]
                if next_segment.get('start_time', 0) - current_segment.get('end_time', 0) > 0.2:
                    # There's a gap, so clear the subtitle during this gap
                    if current_time > current_segment.get('end_time', 0):
                        return "", None
        
        return text, None
    
    return default_text, None

async def process_debate_segments(segments: List[Dict[str, str]], output_dir: str = 'outputs/audio_output') -> bool:
    """Process debate segments and generate audio."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean output directory before starting, but do it safely
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                try:
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                            print(f"Removed old file: {file}")
                        except PermissionError:
                            print(f"Warning: Could not remove file in use: {file} - will use a different filename")
                        except Exception as e:
                            print(f"Warning: Could not remove old file {file}: {e}")
                except Exception as e:
                    print(f"Warning: Error accessing file {file}: {e}")
        
        success_count = 0
        for i, segment in enumerate(segments):
            speaker = segment["speaker"]
            text = segment["text"]
            
            # Ensure we capture complete text, even without labels
            text = text.strip()
            
            # Select voice based on speaker using integer ID mapping
            voice = VOICES.get(speaker, VOICES['Narrator'])
            
            # Create audio file with index-based naming
            # If file is in use, the text_to_speech function will handle creating an alternate name
            output_file = f'{output_dir}/part_{i:02d}.wav'
            
            print(f"Generating audio for segment {i+1}/{len(segments)} - Speaker: {speaker} (Voice ID: {voice})")
            
            # Process segments with retries
            success = await text_to_speech(text, voice, output_file, max_retries=3)
            
            if success:
                success_count += 1
                print(f"Successfully created audio for segment {i+1}")
            else:
                print(f"Failed to create audio for segment {i+1} after retries")
        
        print(f"Successfully created {success_count}/{len(segments)} audio segments")
        return success_count > 0
    except Exception as e:
        print(f"Error in process_debate_segments: {e}")
        import traceback
        print(traceback.format_exc())
        return False

async def generate_debate_speech(segments: List[Dict[str, str]], 
                               output_dir: str = 'outputs/audio_output') -> bool:
    """Generate speech for all debate segments."""
    try:
        success = await process_debate_segments(segments, output_dir)
        return success
    except Exception as e:
        print(f"Error generating debate speech: {e}")
        return False

def clean_sentence(text):
    """Clean up a sentence for better subtitle display.
    
    Args:
        text: The text to clean
        
    Returns:
        The cleaned text
    """
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Ensure text doesn't start with punctuation (common in incomplete sentences)
    while text and text[0] in ',.;:!?':
        text = text[1:].lstrip()
    
    # Capitalize first letter if it's a new sentence 
    if text and not text[0].isupper() and not text[0].isdigit():
        text = text[0].upper() + text[1:]
    
    return text

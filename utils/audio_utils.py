import os
import json
from pydub import AudioSegment
import edge_tts
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from utils.text_utils import split_text_into_smaller_parts

def get_segment_audio_file(index):
    """Get the audio file for a specific segment."""
    try:
        # Just use the index directly for predictable file naming
        expected_file = os.path.join('outputs/audio_output', f'part_{index:02d}.mp3')
        
        if os.path.exists(expected_file) and os.path.getsize(expected_file) > 0:
            return expected_file
            
        # Fallback to old method of searching files if direct naming fails
        files = [f for f in os.listdir('outputs/audio_output') if f.endswith('.mp3') and not f.endswith('_timing.mp3')]
        files.sort()
        
        if index < len(files):
            return os.path.join('outputs/audio_output', files[index])
        else:
            print(f"Warning: No audio file found for segment {index} (total files: {len(files)})")
            return None
    except Exception as e:
        print(f"Error finding audio file: {str(e)}")
        return None

def get_segment_duration(audio_file):
    """Get the duration of an audio segment."""
    if not audio_file or not os.path.exists(audio_file):
        return 5.0  # Default duration if no audio file
        
    try:
        # Use pydub which is more reliable than ffmpeg directly
        audio = AudioSegment.from_file(audio_file)
        duration = len(audio) / 1000.0
        return duration
    except Exception as e:
        print(f"Error getting audio duration: {str(e)}")
        print(f"Using default duration for {audio_file}")
        return 5.0  # Default duration on error

def get_all_timing_data():
    """Get all timing data across all segments for a comprehensive view."""
    all_timing = []
    
    try:
        # Get all timing JSON files
        timing_files = [f for f in os.listdir('outputs/audio_output') if f.endswith('_timing.json')]
        timing_files.sort()  # Ensure proper order
        
        # Absolute start time tracker to adjust all segments to a global timeline
        absolute_start = 0
        
        for timing_file in timing_files:
            try:
                # Extract the audio file name for this timing file
                audio_file = timing_file.replace('_timing.json', '.mp3')
                audio_path = os.path.join('outputs/audio_output', audio_file)
                
                # Get the duration of this audio segment
                duration = 0
                if os.path.exists(audio_path):
                    audio = AudioSegment.from_mp3(audio_path)
                    duration = len(audio) / 1000.0
                
                # Load timing data
                with open(os.path.join('outputs/audio_output', timing_file), 'r') as f:
                    timing_data = json.load(f)
                
                # Get speaker for this segment
                speaker = None
                if timing_data.get("segments") and len(timing_data["segments"]) > 0:
                    first_segment = timing_data["segments"][0]
                    if "text" in first_segment:
                        # Try to determine speaker from text
                        text = first_segment["text"]
                        speaker = get_speaker_for_text(text)
                
                # Process segments and adjust to absolute timeline
                for segment in timing_data.get("segments", []):
                    # Make a copy with adjusted timing
                    adjusted_segment = segment.copy()
                    
                    if "start_time" in adjusted_segment:
                        adjusted_segment["start_time"] += absolute_start
                    elif "start" in adjusted_segment:
                        adjusted_segment["start_time"] = adjusted_segment["start"] + absolute_start
                        adjusted_segment["start"] += absolute_start
                        
                    if "end_time" in adjusted_segment:
                        adjusted_segment["end_time"] += absolute_start
                    elif "end" in adjusted_segment:
                        adjusted_segment["end_time"] = adjusted_segment["end"] + absolute_start
                        adjusted_segment["end"] += absolute_start
                    
                    # Add speaker information if available
                    if speaker:
                        adjusted_segment["speaker"] = speaker
                    else:
                        # Try to determine speaker from text
                        if "text" in adjusted_segment:
                            adjusted_segment["speaker"] = get_speaker_for_text(adjusted_segment["text"])
                    
                    all_timing.append(adjusted_segment)
                
                # Update absolute start time for next segment
                absolute_start += duration
                
            except Exception as e:
                print(f"Error processing timing file {timing_file}: {str(e)}")
        
        print(f"Loaded {len(all_timing)} timing segments across {len(timing_files)} audio files")
        return all_timing
        
    except Exception as e:
        print(f"Error getting timing data: {str(e)}")
        return []

def get_speaker_for_text(text):
    """Determine the speaker for a given text segment."""
    global _speaker_map_cache
    
    if _speaker_map_cache is None:
        _speaker_map_cache = parse_debate_file()
    
    # Try exact match first
    if text in _speaker_map_cache:
        return _speaker_map_cache[text]
    
    # Try to find a partial match
    best_match = None
    best_match_length = 0
    best_match_speaker = None
    
    for mapped_text, speaker in _speaker_map_cache.items():
        # Check if text is in the mapped text
        if text in mapped_text and len(mapped_text) > best_match_length:
            best_match = text
            best_match_length = len(mapped_text)
            best_match_speaker = speaker
        # Check if mapped text is in the text
        elif mapped_text in text and len(mapped_text) > best_match_length:
            best_match = mapped_text
            best_match_length = len(mapped_text)
            best_match_speaker = speaker
    
    if best_match_speaker:
        return best_match_speaker
    
    # Default to Narrator if no match found
    return "Narrator"

def get_segment_timing(audio_file):
    """Get the timing information for a specific audio segment."""
    # First, try to get all comprehensive timing data
    all_timing = getattr(get_segment_timing, 'all_timing', None)
    if all_timing is None:
        all_timing = get_all_timing_data()
        # Cache it for future calls
        get_segment_timing.all_timing = all_timing
    
    if not audio_file:
        return all_timing
    
    # If we need specific segment timing, get the file's timing
    timing_file = audio_file.replace('.mp3', '_timing.json')
    try:
        if os.path.exists(timing_file):
            with open(timing_file, 'r') as f:
                timing_data = json.load(f)
                segments = timing_data.get("segments", [])
                # Process each segment to ensure it has required fields
                for segment in segments:
                    if isinstance(segment, dict):
                        # Ensure timing fields exist
                        if "start_time" not in segment:
                            segment["start_time"] = segment.get("start", 0)
                        if "end_time" not in segment:
                            segment["end_time"] = segment.get("end", 5)
                        
                        # Extract text if present but not in expected field
                        if "text" not in segment and "words" in segment:
                            segment["text"] = " ".join([w.get("word", "") for w in segment["words"]])
                        
                        # Add speaker information
                        if "text" in segment and "speaker" not in segment:
                            segment["speaker"] = get_speaker_for_text(segment["text"])
                
                return segments
        else:
            # Return the relevant portion of the global timing data
            return all_timing
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read timing file for {audio_file}: {str(e)}")
        # Fall back to global timing data
        return all_timing

def parse_debate_file():
    """Parse the debate.txt file to map text segments to speakers."""
    speaker_text_map = {}
    try:
        with open('outputs/debate.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split by line breaks and process
        lines = content.split('\n')
        current_speaker = "Narrator"
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip Display Summary line - it should only be displayed, not spoken
            if line.startswith("Display Summary:"):
                continue
                
            # Check for speaker patterns
            if line.startswith("Ground Statement:") or line.startswith("Result:"):
                # Complete previous speaker's text
                if current_text:
                    full_text = ' '.join(current_text).strip()
                    # Add the full text first for better matching
                    if full_text:
                        speaker_text_map[full_text] = current_speaker
                    # Then add individual sentences for partial matching
                    for sentence in full_text.replace('! ', '!<SPLIT>').replace('? ', '?<SPLIT>').replace('. ', '.<SPLIT>').split('<SPLIT>'):
                        if sentence.strip():
                            speaker_text_map[sentence.strip()] = current_speaker
                
                current_speaker = "Narrator"
                current_text = [line]
            elif ":" in line:
                speaker_part = line.split(":", 1)[0].strip()
                # Check for valid speaker labels
                if speaker_part in ["Jane", "Valentino", "AI Debater 1", "AI Debater 2", "Narrator"]:
                    # Complete previous speaker's text
                    if current_text:
                        full_text = ' '.join(current_text).strip()
                        # Add the full text first for better matching
                        if full_text:
                            speaker_text_map[full_text] = current_speaker
                        # Then add individual sentences for partial matching
                        for sentence in full_text.replace('! ', '!<SPLIT>').replace('? ', '?<SPLIT>').replace('. ', '.<SPLIT>').split('<SPLIT>'):
                            if sentence.strip():
                                speaker_text_map[sentence.strip()] = current_speaker
                    
                    # Map AI Debater 1/2 to Jane/Valentino if needed
                    if speaker_part == "AI Debater 1":
                        current_speaker = "Jane"
                    elif speaker_part == "AI Debater 2":
                        current_speaker = "Valentino"
                    else:
                        current_speaker = speaker_part
                    
                    current_text = [line.split(":", 1)[1].strip()]
                else:
                    # If not a recognized speaker label, continue with current text
                    current_text.append(line)
            else:
                current_text.append(line)
        
        # Add final text segment
        if current_text:
            full_text = ' '.join(current_text).strip()
            # Add the full text first for better matching
            if full_text:
                speaker_text_map[full_text] = current_speaker
            # Then add individual sentences for partial matching
            for sentence in full_text.replace('! ', '!<SPLIT>').replace('? ', '?<SPLIT>').replace('. ', '.<SPLIT>').split('<SPLIT>'):
                if sentence.strip():
                    speaker_text_map[sentence.strip()] = current_speaker
        
        return speaker_text_map
    except Exception as e:
        print(f"Error parsing debate file: {str(e)}")
        return {}

# Cache for the speaker map to avoid repeated parsing
_speaker_map_cache = None

def get_current_subtitle(timing_segments, current_time, original_text) -> Tuple[str, Optional[str]]:
    """
    Get the subtitle text and speaker that should be displayed at the current time.
    
    Returns:
        Tuple containing (subtitle_text, speaker_name)
    """
    global _speaker_map_cache
    
    if not timing_segments:
        return original_text, None
    
    # Initialize speaker map if not already done
    if _speaker_map_cache is None:
        _speaker_map_cache = parse_debate_file()
    
    # Find active segments for the current time
    active_segments = []
    for segment in timing_segments:
        start_time = segment.get("start_time", segment.get("start", 0))
        end_time = segment.get("end_time", segment.get("end", 5))
        
        if start_time <= current_time <= end_time:
            active_segments.append(segment)
    
    # If we have multiple active segments, choose the one with the longest text
    if active_segments:
        best_segment = max(active_segments, key=lambda s: len(s.get("text", "")))
        segment_text = best_segment.get("text", "").strip()
        
        # First check if the segment already has speaker information
        if "speaker" in best_segment and best_segment["speaker"]:
            return segment_text, best_segment["speaker"]
        
        if segment_text:
            # Look for an exact match in our speaker map
            if segment_text in _speaker_map_cache:
                current_speaker = _speaker_map_cache[segment_text]
                return segment_text, current_speaker
            
            # Try to find a partial match
            best_match_speaker = get_speaker_for_text(segment_text)
            if best_match_speaker:
                return segment_text, best_match_speaker
    
    # If no specific segment text is found, return empty
    return "", None

# Voice settings
VOICES = {
    'Narrator': 'en-US-ChristopherNeural',
    'Jane': 'en-US-JennyNeural',
    'Valentino': 'en-GB-RyanNeural'
}

async def text_to_speech(text: str, voice: str, output_file: str) -> bool:
    """Convert text to speech using Edge TTS with detailed timing information."""
    try:
        # Create communicate object
        communicate = edge_tts.Communicate(text, voice)
        
        # Create both the audio file and timing file simultaneously
        timing_data = {"segments": []}
        word_boundary_list = []
        
        # Generate audio - don't use async with for BufferedWriter
        audio_file = open(output_file, "wb")
        try:
            # Process the audio stream with timing information
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    word_boundary_list.append(chunk)
        finally:
            audio_file.close()
        
        print(f"Created audio file: {output_file}")
        
        # Verify the audio file was created and is valid
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            print(f"Error: Failed to create valid audio file at {output_file}")
            return False
            
        # Calculate the duration of the audio file
        audio_duration = 0
        try:
            audio = AudioSegment.from_file(output_file)
            audio_duration = len(audio) / 1000.0
            print(f"Audio duration: {audio_duration:.2f}s")
        except Exception as e:
            print(f"Warning: Could not determine audio duration: {e}")
            audio_duration = 30.0  # Default assumption
        
        # Create simple timing segments without relying on word boundaries
        # Split text into smaller chunks for better readability
        chunks = split_text_into_smaller_parts(text)
        chunk_duration = audio_duration / max(len(chunks), 1)
        
        # Create timing segments based on text chunks
        for i, chunk_text in enumerate(chunks):
            start_time = i * chunk_duration
            end_time = (i + 1) * chunk_duration
            
            # Ensure the last segment ends exactly at the audio duration
            if i == len(chunks) - 1:
                end_time = audio_duration
            
            segment = {
                "text": chunk_text,
                "start_time": start_time,
                "end_time": end_time
            }
            timing_data["segments"].append(segment)
        
        # Save timing data to file
        timing_file = output_file.replace('.mp3', '_timing.json')
        with open(timing_file, 'w') as f:
            json.dump(timing_data, f, indent=2)
            print(f"Created timing file with {len(timing_data['segments'])} segments: {timing_file}")
        
        return True
    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        return False

async def process_debate_segments(segments: List[Dict[str, str]], output_dir: str = 'outputs/audio_output') -> bool:
    """Process debate segments and generate audio."""
    try:
        # Clean output directory before starting
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                try:
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not remove old file {file}: {e}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        success_count = 0
        for i, segment in enumerate(segments):
            speaker = segment["speaker"]
            text = segment["text"]
            
            # Select voice based on speaker
            voice = VOICES.get(speaker, VOICES['Narrator'])
            
            # Create audio file
            output_file = f'{output_dir}/part_{i:02d}.mp3'
            print(f"Generating audio for segment {i+1}/{len(segments)} - Speaker: {speaker}")
            
            success = await text_to_speech(text, voice, output_file)
            
            if success:
                success_count += 1
            else:
                print(f"Failed to create audio for segment {i+1}")
                # Create an empty placeholder file to maintain sequence numbering
                with open(output_file, 'wb') as f:
                    # Write a minimal valid MP3 header - 3 seconds of silence
                    silent_mp3 = AudioSegment.silent(duration=3000)
                    silent_mp3.export(f, format="mp3")
                print(f"Created placeholder silent audio for segment {i+1}")
                
                # Create a basic timing file
                timing_file = output_file.replace('.mp3', '_timing.json')
                with open(timing_file, 'w') as f:
                    json.dump({
                        "segments": [{
                            "text": text,
                            "start_time": 0,
                            "end_time": 3.0
                        }]
                    }, f, indent=2)
        
        print(f"Successfully created {success_count}/{len(segments)} audio segments")
        return success_count > 0
    except Exception as e:
        print(f"Error in process_debate_segments: {e}")
        return False

async def generate_debate_speech(segments: List[Dict[str, str]], 
                               output_dir: str = 'outputs/audio_output') -> bool:
    """Generate speech for all debate segments."""
    try:
        success = await process_debate_segments(segments, output_dir)
        return success
    except Exception as e:
        print(f"Error in generate_debate_speech: {e}")
        return False

import json
import os
from typing import List, Dict

import edge_tts
from pydub import AudioSegment

from utils.text_utils import split_text_into_chunks

# Define voices
VOICES = {
    "Narrator": "en-US-ChristopherNeural",
    "AI Debater 1": "en-GB-RyanNeural",
    "AI Debater 2": "en-US-JasonNeural",
    "Jane": "en-GB-SoniaNeural",
    "Valentino": "en-US-GuyNeural"
}

def get_segment_audio_file(segment_index):
    """Get the audio file for a specific segment index."""
    try:
        audio_file = f'outputs/audio_output/part_{segment_index:02d}.mp3'
        if os.path.exists(audio_file):
            return audio_file
        
        # Try to find by looking at available files
        files = [f for f in os.listdir('outputs/audio_output') if f.endswith('.mp3') and not f.endswith('_timing.mp3')]
        files.sort()
        if segment_index < len(files):
            return os.path.join('outputs/audio_output', files[segment_index])
        
        return None
    except Exception as e:
        print(f"Error getting segment audio file: {str(e)}")
        return None

def get_segment_duration(audio_file):
    """Get the duration of an audio file in seconds."""
    try:
        if audio_file and os.path.exists(audio_file):
            audio = AudioSegment.from_mp3(audio_file)
            return len(audio) / 1000.0  # pydub uses milliseconds
        return 5.0  # Default duration if we can't determine
    except Exception as e:
        print(f"Error getting audio duration: {str(e)}")
        return 5.0

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
                audio_file = timing_file.replace('_timing.json', '.mp3')
                
                # Get audio duration
                audio_duration = 0.0
                if os.path.exists(os.path.join('outputs/audio_output', audio_file)):
                    audio = AudioSegment.from_mp3(os.path.join('outputs/audio_output', audio_file))
                    audio_duration = len(audio) / 1000.0
                
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
        
    # If we need specific segment timing, get the file's timing
    timing_file = audio_file.replace('.mp3', '_timing.json')
    
    try:
        if os.path.exists(timing_file):
            with open(timing_file, 'r') as f:
                timing_data = json.load(f)
                segments = timing_data.get("segments", [])
                
                # Ensure all segments have proper timing fields
                for segment in segments:
                    if "start_time" not in segment:
                        segment["start_time"] = 0.0
                    if "end_time" not in segment:
                        segment["end_time"] = get_segment_duration(audio_file)
                        
                return segments
        else:
            # Return the relevant portion of the global timing data
            return all_timing
    except Exception as e:
        print(f"Warning: Could not read timing file for {audio_file}: {str(e)}")
        # Fall back to global timing data
        return all_timing

async def text_to_speech(text: str, voice: str, output_file: str) -> bool:
    """Convert text to speech using Edge TTS with detailed timing information."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create both the audio file and timing file simultaneously
        timing_data = {"segments": []}
        
        print(f"Creating audio file: {output_file}")
        
        try:
            # Initialize Edge TTS
            communicate = edge_tts.Communicate(text, voice)
            
            # Process the audio stream with timing information
            await communicate.save(output_file)
            
            # Get duration of the audio file
            duration = 0
            try:
                audio = AudioSegment.from_mp3(output_file)
                duration = len(audio) / 1000.0  # Convert ms to seconds
                print(f"Audio duration: {duration:.2f}s")
            except Exception as e:
                print(f"Warning: Could not get audio duration: {str(e)}")
                duration = 5.0  # Default duration if we can't determine it
                
        except Exception as e:
            print(f"Error in TTS generation: {str(e)}")
            return False
        
        # Create timing segments based on improved chunking
        try:
            # Smaller chunk size for better readability (30-40 chars per line)
            max_chunk_size = 80

            # Split the entire text into smaller, more readable chunks
            text_chunks = split_text_into_chunks(text, max_chars=max_chunk_size)
            
            if len(text_chunks) == 1:
                # For very short text, use a single segment
                segment = {
                    "text": text,
                    "start_time": 0,
                    "end_time": duration
                }
                timing_data["segments"] = [segment]
            else:
                # Calculate timing for each chunk
                chunk_duration = duration / len(text_chunks)
                
                # Create timing segments based on text chunks
                for i, chunk in enumerate(text_chunks):
                    start_time = i * chunk_duration
                    end_time = (i + 1) * chunk_duration
                    
                    segment = {
                        "text": chunk,
                        "start_time": start_time,
                        "end_time": end_time
                    }
                    timing_data["segments"].append(segment)
        except Exception as e:
            print(f"Warning: Could not create detailed timing: {str(e)}")
            # Create at least one basic timing segment
            timing_data["segments"] = [{
                "text": text,
                "start_time": 0,
                "end_time": duration
            }]
        
        # Save timing data to file
        timing_file = output_file.replace('.mp3', '_timing.json')
        with open(timing_file, 'w') as f:
            json.dump(timing_data, f, indent=2)
            print(f"Created timing file with {len(timing_data['segments'])} segments: {timing_file}")
        
        return True
    except Exception as e:
        print(f"Error in text_to_sech: {str(e)}")
        return False

def get_current_subtitle(timing_segments, current_time, default_text=''):
    """Get the current subtitle text based on timing information.
    
    Args:
        timing_segments: List of timing segments with text and timestamps
        current_time: Current time in the audio playback
        default_text: Default text if no matching segment is found
        
    Returns:
        Tuple of (current_subtitle_text, speaker_name)
    """
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
            
            # Ensure we capture complete text, even without labels
            # This will help prevent missing text like "Welcome to our AI debate"
            text = text.strip()
            
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
        print(f"Error generating debate speech: {e}")
        return False

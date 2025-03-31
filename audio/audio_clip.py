import json
import os
from pydub import AudioSegment
from utils.text_utils import split_text_into_chunks

class AudioClip:
    def __init__(self, file_path=None, timing_data=None, segment_index=None):
        self.file_path = file_path
        self.timing_data = timing_data or {"segments": []}
        self.segment_index = segment_index
        self._duration = None
        
        # Load timing data if file path is provided
        if self.file_path and os.path.exists(self.file_path):
            self.load_timing_data()
    
    @property
    def duration(self):
        """Get the duration of the audio clip in seconds."""
        if self._duration is None and self.file_path and os.path.exists(self.file_path):
            try:
                audio = AudioSegment.from_mp3(self.file_path)
                self._duration = len(audio) / 1000.0  # pydub uses milliseconds
            except Exception as e:
                print(f"Error getting audio duration: {str(e)}")
                self._duration = 5.0  # Default duration if we can't determine
        return self._duration or 5.0
    
    @classmethod
    def from_segment_index(cls, segment_index):
        """Create an AudioClip from a segment index."""
        try:
            audio_file = f'outputs/audio_output/part_{segment_index:02d}.mp3'
            if os.path.exists(audio_file):
                return cls(audio_file, segment_index=segment_index)
            
            # Try to find by looking at available files
            files = [f for f in os.listdir('outputs/audio_output') if f.endswith('.mp3') and not f.endswith('_timing.mp3')]
            files.sort()
            if segment_index < len(files):
                return cls(os.path.join('outputs/audio_output', files[segment_index]), segment_index=segment_index)
            
            return None
        except Exception as e:
            print(f"Error getting segment audio file: {str(e)}")
            return None
    
    def load_timing_data(self):
        """Load timing data for this audio clip."""
        if not self.file_path:
            return
            
        timing_file = self.file_path.replace('.mp3', '_timing.json')
        
        try:
            if os.path.exists(timing_file):
                with open(timing_file, 'r') as f:
                    self.timing_data = json.load(f)
                    segments = self.timing_data.get("segments", [])
                    
                    # Ensure all segments have proper timing fields
                    for segment in segments:
                        if "start_time" not in segment:
                            segment["start_time"] = 0.0
                        if "end_time" not in segment:
                            segment["end_time"] = self.duration
        except Exception as e:
            print(f"Warning: Could not read timing file for {self.file_path}: {str(e)}")
    
    def get_current_subtitle(self, current_time, default_text=''):
        """Get the current subtitle text based on timing information."""
        if not self.timing_data or not self.timing_data.get("segments"):
            return default_text, None
        
        timing_segments = self.timing_data.get("segments", [])
        
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
            
    @staticmethod
    def mix_with_background_music(audio_file, bg_music_file="assets/background_music.mp3", bg_volume=0.15):
        """
        Mix the audio file with background music at a reduced volume.
        
        Args:
            audio_file: Path to the audio file to mix
            bg_music_file: Path to the background music file
            bg_volume: Volume level for background music (0.0 to 1.0)
            
        Returns:
            Path to the mixed audio file
        """
        try:
            if not os.path.exists(bg_music_file):
                print(f"Background music file not found: {bg_music_file}")
                return audio_file
                
            # Load the audio tracks
            audio = AudioSegment.from_file(audio_file)
            
            # Load background music
            bg_music = AudioSegment.from_file(bg_music_file)
            
            # Adjust background music volume (reduce it to be subtle)
            bg_music = bg_music - (1 - bg_volume) * 20
            
            # Loop background music if it's shorter than speech
            if len(bg_music) < len(audio):
                # Calculate how many loops we need
                loops_needed = len(audio) // len(bg_music) + 1
                bg_music = bg_music * loops_needed
            
            # Trim background music to match audio length
            bg_music = bg_music[:len(audio)]
            
            # Mix audio with background music
            mixed_audio = audio.overlay(bg_music)
            
            # Create the output file
            output_file = audio_file.replace(".mp3", "_with_bg.mp3")
            mixed_audio.export(output_file, format="mp3")
            
            return output_file
        except Exception as e:
            print(f"Error mixing background music: {str(e)}")
            return audio_file  # Return original file if mixing fails

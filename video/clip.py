from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.audio.AudioClip import AudioClip, concatenate_audioclips

class VideoClip:
    """A wrapper class for managing video clips and their properties."""
    
    def __init__(self, clip, index=None):
        """
        Initialize a VideoClip object.
        
        Args:
            clip: MoviePy VideoClip object
            index: Optional identifier for the clip
        """
        self.clip = clip
        self.index = index
    
    def validate_audio(self):
        """Validate and fix audio in clip to prevent buffer boundary issues."""
        try:
            if self.clip.audio is None:
                print(f"Warning: Clip {self.index} has no audio, creating silent audio")
                self.clip = self.clip.set_audio(AudioClip(lambda t: 0, duration=self.clip.duration))
                return self
            
            # Ensure audio duration matches or is slightly shorter than video
            if abs(self.clip.audio.duration - self.clip.duration) > 0.1:
                print(f"Warning: Clip {self.index} audio/video duration mismatch: "
                      f"Audio={self.clip.audio.duration:.2f}s, Video={self.clip.duration:.2f}s")
                
                # Instead of trimming audio, extend the video duration to fit the audio
                if self.clip.audio.duration > self.clip.duration:
                    print(f"Extending video duration to match audio for clip {self.index}")
                    # Create a copy of the clip with the last frame extended
                    last_frame = self.clip.get_frame(self.clip.duration - 0.01)
                    extension = ImageClip(last_frame).set_duration(self.clip.audio.duration - self.clip.duration)
                    
                    # Optimization: Use 'compose' method for faster concatenation
                    extended_clip = concatenate_videoclips(
                        [self.clip, extension], 
                        method='compose',
                        bg_color=None, 
                        use_bgclip=False
                    )
                    extended_clip = extended_clip.set_audio(self.clip.audio)
                    print(f"Extended video from {self.clip.duration:.2f}s to {extended_clip.duration:.2f}s")
                    self.clip = extended_clip
                else:
                    # If audio is too short, extend with silence
                    silence = AudioClip(lambda t: 0, duration=self.clip.duration - self.clip.audio.duration)
                    self.clip = self.clip.set_audio(concatenate_audioclips([self.clip.audio, silence]))
                    print(f"Extended audio with silence for clip {self.index}")
            
            return self
        except Exception as e:
            print(f"Error validating clip {self.index} audio: {str(e)}")
            return self
    
    def resize(self, width=None, height=None):
        """Resize the clip to the specified dimensions."""
        if width or height:
            self.clip = self.clip.resize(width=width, height=height)
        return self
    
    def set_position(self, position):
        """Set the position of the clip in the composition."""
        self.clip = self.clip.set_position(position)
        return self
    
    def set_duration(self, duration):
        """Set the duration of the clip."""
        if duration != self.clip.duration:
            self.clip = self.clip.set_duration(duration)
        return self
    
    def subclip(self, start_time, end_time):
        """Extract a portion of the clip between the specified times."""
        self.clip = self.clip.subclip(start_time, end_time)
        return self
    
    def set_audio(self, audio_clip):
        """Set or replace the audio for this clip."""
        self.clip = self.clip.set_audio(audio_clip)
        return self
    
    def get_raw_clip(self):
        """Get the underlying MoviePy clip object."""
        return self.clip
    
    @staticmethod
    def concatenate(clips, method='compose'):
        """
        Concatenate multiple VideoClip objects with optimized settings.
        
        Args:
            clips: List of VideoClip objects to concatenate
            method: Method for concatenation ('compose' is faster than 'chain')
        
        Returns:
            A new VideoClip object with the concatenated clip
        """
        if not clips:
            return None
            
        if len(clips) == 1:
            return clips[0]
        
        # Extract raw MoviePy clips
        raw_clips = [clip.get_raw_clip() for clip in clips]
        
        try:
            # Performance optimizations for concatenation
            result = concatenate_videoclips(
                raw_clips,
                method=method,
                bg_color=None,
                use_bgclip=False
            )
            return VideoClip(result)
        except Exception as e:
            print(f"Error in fast concatenation: {str(e)}")
            # Fallback to default concatenation
            result = concatenate_videoclips(raw_clips)
            return VideoClip(result)

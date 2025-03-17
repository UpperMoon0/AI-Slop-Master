import os
import gc
import multiprocessing as mp

from utils.file_utils import parse_debate_file, cleanup_temp_files
from utils.audio_utils import get_segment_audio_file
from utils.video_utils import create_segment_video, combine_video_segments
from config import TEMP_FRAMES_DIR, PROJECT_TEMP_DIR

def create_debate_video():
    """Create a video visualization of the debate with audio."""
    # Set process start method for Windows
    if os.name == 'nt':
        mp.set_start_method('spawn', force=True)
    
    # Parse the debate file
    dialogue_segments = parse_debate_file()
    if not dialogue_segments:
        print("No dialogue segments found. Aborting video creation.")
        return
    
    try:
        # Process segments sequentially and create clips
        segment_clips = []
        for i, segment in enumerate(dialogue_segments):
            speaker = segment["speaker"]
            text = segment["text"]
            
            # Get audio file for this segment
            audio_file = get_segment_audio_file(i)
            if not audio_file or not os.path.exists(audio_file):
                print(f"Warning: No audio file found for segment {i}")
                continue
            
            # Create video clip for this segment
            clip = create_segment_video(i, speaker, text, audio_file)
            if clip:
                segment_clips.append(clip)
        
        # Combine all segment clips into final video
        if segment_clips:
            combine_video_segments(segment_clips, "outputs/debate.mp4")
        else:
            print("No clips were created. Cannot generate final video.")
    
    except Exception as e:
        print(f"Error during video creation: {str(e)}")
    
    finally:
        # Give some time for all file handles to be released before cleanup
        import time
        time.sleep(2)
        
        # Clean up temporary files
        cleanup_temp_files(TEMP_FRAMES_DIR, PROJECT_TEMP_DIR)
        
        # Force garbage collection
        gc.collect()

if __name__ == "__main__":
    create_debate_video()
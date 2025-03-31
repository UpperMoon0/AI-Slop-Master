import os
import gc
import time
import multiprocessing as mp

from moviepy import VideoFileClip
from tqdm import tqdm


from utils.file_utils import parse_debate_file, cleanup_temp_files
from utils.audio_utils import get_segment_audio_file
from utils.video_utils import create_segment_video, combine_video_segments
from config import TEMP_FRAMES_DIR, PROJECT_TEMP_DIR

def create_debate_video(output_path='outputs/debate.mp4', mode='fast', batch_size=30, 
                       add_bg_music=True, bg_music_file="assets/background_music.mp3", bg_volume=0.15):
    """Create a video visualization of the debate with audio.
    
    Args:
        output_path: Path where the final video will be saved
        mode: 'fast' (fastest, lower quality) or 'slow' (best quality, slower)
        batch_size: Number of clips to process in each batch for memory efficiency
        add_bg_music: Whether to add background music
        bg_music_file: Path to background music file
        bg_volume: Volume level for background music (0.0 to 1.0)
    """
    print("\n=== Starting Video Generation Process ===")
    start_time = time.time()
    
    # Set process start method for Windows
    if os.name == 'nt':
        mp.set_start_method('spawn', force=True)
    
    # Parse the debate file
    print("Step 1: Parsing dialogue segments...")
    segment_start = time.time()
    dialogue_segments = parse_debate_file()
    print(f"  √ Parsed {len(dialogue_segments)} dialogue segments in {time.time() - segment_start:.2f} seconds")
    
    if not dialogue_segments:
        print("No dialogue segments found. Aborting video creation.")
        return
    
    try:
        # Check if background music exists
        if add_bg_music and not os.path.exists(bg_music_file):
            print(f"Warning: Background music file not found: {bg_music_file}")
            print("Background music will be disabled.")
            add_bg_music = False
        
        # Process segments sequentially and create clips
        print("\nStep 2: Generating video clips...")
        clips_start = time.time()
        segment_clips = []
        for i, segment in enumerate(tqdm(dialogue_segments, desc="Creating video segments")):
            speaker = segment["speaker"]
            text = segment["text"]
            
            print(f"  - Processing segment {i+1}/{len(dialogue_segments)}: {speaker}")
            clip_generation_start = time.time()
            
            # Get audio file for this segment
            audio_file = get_segment_audio_file(i)
            if not audio_file or not os.path.exists(audio_file):
                print(f"Warning: No audio file found for segment {i}")
                continue
            
            # Create video clip for this segment with temp_dir
            clip = create_segment_video(i, speaker, text, audio_file, mode=mode, temp_dir=PROJECT_TEMP_DIR)
            if clip:
                segment_clips.append(clip)
            
            # Process in batches to save memory
            if len(segment_clips) >= batch_size:
                batch_output = os.path.join(PROJECT_TEMP_DIR, f"batch_{i//batch_size}.mp4")
                print(f"  - Processing batch {i//batch_size + 1}...")
                combine_video_segments(segment_clips, batch_output, mode=mode, temp_dir=PROJECT_TEMP_DIR)
                segment_clips = [VideoFileClip(batch_output)]
                gc.collect()  # Force garbage collection to free memory
            
            if (i + 1) % 5 == 0 or i == len(dialogue_segments) - 1:
                print(f"  - Video progress: {i+1}/{len(dialogue_segments)} segments ({((i+1)/len(dialogue_segments))*100:.1f}%)")
                print(f"  - Elapsed time: {time.time() - start_time:.2f} seconds")
        
        print(f"  √ Generated {len(dialogue_segments)} video segments in {time.time() - clips_start:.2f} seconds")
        
        # Combine all segment clips into final video
        print("\nStep 3: Concatenating video clips...")
        concat_start = time.time()
        
        quality_descriptions = {
            'fast': "lower quality (fastest processing)",
            'slow': "best quality (standard)"
        }
        print(f"  - This step may take several minutes depending on video length and complexity")
        print(f"  - Generating {quality_descriptions.get(mode, 'standard quality')} video")
        if add_bg_music:
            print(f"  - Adding background music from {bg_music_file}")
        
        if segment_clips:
            combine_video_segments(
                segment_clips, 
                output_path, 
                mode=mode, 
                temp_dir=PROJECT_TEMP_DIR,
                add_bg_music=add_bg_music,
                bg_music_file=bg_music_file,
                bg_volume=bg_volume
            )
            print(f"  √ Concatenated clips in {time.time() - concat_start:.2f} seconds")
        else:
            print("No clips were created. Cannot generate final video.")
    
    except Exception as e:
        print(f"Error during video creation: {str(e)}")
    
    finally:
        # Give some time for all file handles to be released before cleanup
        time.sleep(2)
        
        # Clean up temporary files
        cleanup_temp_files(TEMP_FRAMES_DIR, PROJECT_TEMP_DIR)
        
        # Force garbage collection
        gc.collect()
    
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f"\n=== Video Generation Complete ===")
    print(f"Total processing time: {minutes} minutes {seconds:.2f} seconds")
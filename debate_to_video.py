import os
import gc
import multiprocessing as mp
from moviepy.editor import AudioFileClip, ImageSequenceClip, VideoFileClip, concatenate_videoclips

# Import from utility modules
from config import FPS, PROJECT_TEMP_DIR, TEMP_FRAMES_DIR
from utils.text_utils import wrap_text, split_text_into_smaller_parts, split_text_into_chunks
from utils.audio_utils import get_segment_audio_file, get_segment_duration, get_segment_timing, get_current_subtitle
from utils.file_utils import parse_debate_file, cleanup_temp_files
from utils.video_utils import create_frame, create_frame_worker, validate_clip_audio, fix_video_duration

def create_debate_video():
    """Create a video visualization of the debate with audio."""
    dialogue_segments = parse_debate_file()
    if not dialogue_segments:
        print("No dialogue segments found. Aborting video creation.")
        return
    
    # Use more cores for faster processing
    num_cores = max(mp.cpu_count() - 1, 1)  # Use all available cores except one
    pool = mp.Pool(num_cores)
    
    try:
        print(f"Generating video frames using {num_cores} cores...")
        
        all_clips = []
        
        # Process one segment at a time
        for i, segment in enumerate(dialogue_segments):
            speaker = segment["speaker"]
            text = segment["text"]
            
            audio_file = get_segment_audio_file(i)
            if not audio_file or not os.path.exists(audio_file):
                print(f"Warning: No audio file found for segment {i}")
                continue
                
            duration = max(get_segment_duration(audio_file), 3.0)
            timing_segments = get_segment_timing(audio_file)
            
            # Generate more frames for better timing accuracy - at least 1 frame per second
            frames_per_second = 2  # Generate 2 frames per second for smoother subtitles
            total_frames = max(int(duration * frames_per_second), 1)
            
            # Generate frames with evenly distributed timestamps
            frames = []
            timestamps = []
            
            time_step = duration / total_frames
            for j in range(total_frames):
                time_point = j * time_step
                frame = create_frame(speaker, text, True, time_point, duration, timing_segments)
                if frame is not None:
                    frames.append(frame)
                    timestamps.append(time_point)
            
            if not frames:
                print(f"Warning: No frames created for segment {i}. Creating fallback frame.")
                frame = create_frame(speaker, text, True, 0, duration, None)
                if frame is not None:
                    frames.append(frame)
                    timestamps.append(0)
            
            # Create clip from frames with proper durations
            try:
                # Create a clip with the correct frame rate
                clip = ImageSequenceClip(frames, fps=frames_per_second)
                clip = clip.set_duration(duration)
                
                # Add audio
                audio_clip = AudioFileClip(audio_file)
                clip = clip.set_audio(audio_clip)
                
                all_clips.append(clip)
                print(f"Generated clip for segment {i+1}/{len(dialogue_segments)} - {speaker}")
                
                # Force cleanup after each segment to keep memory usage low
                del frames
                gc.collect()
                
            except Exception as e:
                print(f"Error creating clip for segment {i}: {str(e)}")
                try:
                    # Fallback to simple clip for this segment if there's an error
                    print(f"Using fallback clip for segment {i}")
                    fallback_frame = create_frame(speaker, text, True, 0, duration, None)
                    if fallback_frame is not None:
                        clip = ImageSequenceClip([fallback_frame], fps=1)
                        clip = clip.set_duration(duration)
                        audio_clip = AudioFileClip(audio_file)
                        clip = clip.set_audio(audio_clip)
                        all_clips.append(clip)
                except Exception:
                    print(f"Skipping segment {i} due to errors")
        
        if all_clips:
            # Process the final video creation
            try:
                print("\nCombining video segments...")
                output_file = "outputs/debate.mp4"
                
                # Write clips one by one to temp files, then concatenate them
                temp_files = []
                
                for i, clip in enumerate(all_clips):
                    temp_file = os.path.join(TEMP_FRAMES_DIR, f"seg_{i:03d}.mp4")
                    
                    print(f"Writing segment {i+1}/{len(all_clips)}...")
                    try:
                        # Validate clip audio before writing
                        clip = validate_clip_audio(clip, i)
                        
                        # Fix video duration to prevent frame reading issues
                        clip = fix_video_duration(clip, i)
                        
                        # Use a safer method for writing
                        clip.write_videofile(
                            temp_file,
                            fps=FPS,
                            codec='libx264',
                            audio_codec='aac',
                            preset='ultrafast',
                            threads=num_cores,
                            verbose=False,
                            logger=None,
                            temp_audiofile=os.path.join(PROJECT_TEMP_DIR, f"temp_audio_{i}.m4a"),
                            ffmpeg_params=["-avoid_negative_ts", "1"]
                        )
                        temp_files.append(temp_file)
                    except Exception as e:
                        print(f"Error writing segment {i}: {str(e)}")
                    finally:
                        # Ensure clip is closed even if there's an error
                        try:
                            clip.close()
                        except:
                            pass
                
                # Clear all_clips to free memory
                all_clips = []
                gc.collect()
                
                print("Creating final video...")
                # Read back all temp videos
                video_clips = []
                for i, temp_file in enumerate(temp_files):
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        try:
                            # Set a safety buffer when reading files
                            clip = VideoFileClip(temp_file, target_resolution=None, resize_algorithm='fast_bilinear')
                            # Validate audio again after loading from file
                            clip = validate_clip_audio(clip, i)
                            video_clips.append(clip)
                        except Exception as e:
                            print(f"Error loading clip {temp_file}: {str(e)}")
                
                if video_clips:
                    # Try different concatenation methods if needed
                    try:
                        print("Using concatenation method: chain")
                        final_clip = concatenate_videoclips(video_clips, method="chain")
                        
                        final_clip.write_videofile(
                            output_file,
                            fps=FPS,
                            codec='libx264',
                            audio_codec='aac',
                            preset='medium',
                            threads=num_cores,
                            verbose=False,
                            logger=None,
                            temp_audiofile=os.path.join(PROJECT_TEMP_DIR, "temp_final_audio.m4a"),
                            ffmpeg_params=["-avoid_negative_ts", "1"]
                        )
                        
                        print(f"Video saved as: {output_file}")
                        final_clip.close()
                    except Exception as e:
                        print(f"Error with chain concatenation: {str(e)}")
                        
                        try:
                            print("Retrying with different concatenation method: compose")
                            # Close existing clips first
                            for clip in video_clips:
                                try:
                                    clip.close()
                                except:
                                    pass
                            
                            # Reload clips with more conservative settings
                            video_clips = []
                            for i, temp_file in enumerate(temp_files):
                                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                                    try:
                                        clip = VideoFileClip(temp_file, target_resolution=None)
                                        clip = fix_video_duration(clip, i)  # Apply duration fix
                                        video_clips.append(clip)
                                    except Exception as e:
                                        print(f"Error loading clip on retry {temp_file}: {str(e)}")
                            
                            if video_clips:
                                final_clip = concatenate_videoclips(video_clips, method="compose")
                                
                                final_clip.write_videofile(
                                    output_file,
                                    fps=FPS,
                                    codec='libx264',
                                    audio_codec='aac',
                                    preset='medium',
                                    threads=num_cores,
                                    verbose=False,
                                    logger=None,
                                    temp_audiofile=os.path.join(PROJECT_TEMP_DIR, "temp_final_audio_retry.m4a"),
                                    ffmpeg_params=["-avoid_negative_ts", "1"]
                                )
                                
                                print(f"Video saved as: {output_file} (retry method)")
                                final_clip.close()
                        except Exception as e:
                            print(f"Error on retry concatenation: {str(e)}")
                            
                            # Last resort fallback - just use the first clip
                            try:
                                print("Using fallback single-clip method")
                                fallback_file = temp_files[0] if temp_files else None
                                if fallback_file and os.path.exists(fallback_file):
                                    fallback_clip = VideoFileClip(fallback_file)
                                    if os.path.exists("outputs/debate.mp3"):
                                        audio = AudioFileClip("outputs/debate.mp3")
                                        fallback_clip = fallback_clip.set_audio(audio)
                                    
                                    fallback_clip.write_videofile(
                                        output_file,
                                        fps=FPS,
                                        codec='libx264',
                                        audio_codec='aac',
                                        preset='ultrafast',
                                        threads=num_cores,
                                        verbose=False,
                                        logger=None
                                    )
                                    fallback_clip.close()
                                    print(f"Video saved as: {output_file} (fallback method)")
                            except Exception as e:
                                print(f"All methods failed: {str(e)}")
                    
                    # Final cleanup of video clips
                    for clip in video_clips:
                        try:
                            clip.close()
                        except:
                            pass
                else:
                    print("No valid video clips were found.")
                
                # Clean up temp files
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                
            except Exception as e:
                print(f"Error creating final video: {str(e)}")
                
                # Try simpler fallback solution
                try:
                    print("Trying simplified approach...")
                    # Just use the first segment as placeholder and add audio
                    if temp_files and os.path.exists(temp_files[0]):
                        base_clip = VideoFileClip(temp_files[0])
                        full_audio = AudioFileClip("outputs/debate.mp3")
                        
                        # Make the clip as long as the audio
                        base_clip = base_clip.set_duration(full_audio.duration)
                        base_clip = base_clip.set_audio(full_audio)
                        
                        # Write the video
                        base_clip.write_videofile(
                            output_file,
                            fps=FPS,
                            codec='libx264',
                            audio_codec='aac',
                            preset='ultrafast',
                            threads=num_cores,
                            verbose=False,
                            logger=None,
                            temp_audiofile=os.path.join(PROJECT_TEMP_DIR, "temp_fallback_audio.m4a")
                        )
                        
                        print(f"Video saved as: {output_file} (simplified version)")
                        
                        # Clean up
                        base_clip.close()
                        full_audio.close()
                except Exception as e:
                    print(f"Fallback method failed: {str(e)}")
        else:
            print("No clips were created. Cannot generate final video.")
    
    except Exception as e:
        print(f"Error during video creation: {str(e)}")
    
    finally:
        # Clean up resources
        pool.close()
        pool.join()
        
        # Clean up temporary files using utility function
        cleanup_temp_files(TEMP_FRAMES_DIR, PROJECT_TEMP_DIR)
        
        # Force garbage collection
        gc.collect()

if __name__ == "__main__":
    # Set process start method for Windows
    if os.name == 'nt':  # Windows
        mp.set_start_method('spawn', force=True)
    create_debate_video()
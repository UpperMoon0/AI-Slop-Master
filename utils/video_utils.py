import cv2
import numpy as np
from PIL import Image, ImageDraw
import gc
import os
import multiprocessing as mp
from moviepy.editor import AudioFileClip, ImageSequenceClip, VideoFileClip, concatenate_videoclips

from config import VIDEO_WIDTH, VIDEO_HEIGHT, TEXT_COLOR, HIGHLIGHT_COLOR
from config import JANE_AVATAR, VALENTINO_AVATAR, TEXT_FONT, NAME_FONT
from config import FPS, TEMP_FRAMES_DIR, PROJECT_TEMP_DIR
from utils.text_utils import wrap_text, get_font_metrics
from utils.audio_utils import get_current_subtitle, get_segment_duration, get_segment_timing
from video.text import Text
from video.avatar import Avatar
from utils.file_utils import get_ground_statement_summary
from video.clip import VideoClip

# Text containers for top and bottom text
_top_text = Text(position="top", background=True)
_bottom_text = Text(position="bottom", background=False)

# Create avatar instances
# Move avatars up by 50 pixels
avatar_vertical_offset = 50
jane_pos = (100, (VIDEO_HEIGHT - 250 - 50) // 2 - avatar_vertical_offset)
valentino_pos = (VIDEO_WIDTH - 250 - 100, (VIDEO_HEIGHT - 250 - 50) // 2 - avatar_vertical_offset)

_jane_avatar = None
_valentino_avatar = None

# Initialize avatars once loaded
if JANE_AVATAR is not None and VALENTINO_AVATAR is not None:
    _jane_avatar = Avatar(JANE_AVATAR, "Jane", jane_pos)
    _valentino_avatar = Avatar(VALENTINO_AVATAR, "Valentino", valentino_pos)

# Track debate state and text content
_narrator_state = "preDebate"  # "preDebate", "debate", "postDebate"
_ground_statement_text = ""
_ground_statement_summary = ""
_has_seen_first_debater = False  # Flag to track if we've seen the first debater

# Add these global variables to track the current speaker across frames
_last_detected_speaker = None
_speaker_stability_counter = 0

def create_frame(speaker, text, highlighted=False, current_time=0, total_duration=5.0, timing_segments=None):
    """Create a video frame with speakers and text."""
    global _narrator_state, _ground_statement_text, _ground_statement_summary, _has_seen_first_debater
    global _top_text, _bottom_text, _jane_avatar, _valentino_avatar
    global _last_detected_speaker, _speaker_stability_counter
    
    # Create blank frame - using BGR format for consistency with OpenCV
    frame = np.ones((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8) * 240
    
    # Get current subtitle text and the actual speaker from timing
    current_subtitle, current_speaker = get_current_subtitle(timing_segments, current_time, text)
    
    # Update state and text based on current subtitle content
    if current_subtitle:
        if "Ground Statement:" in current_subtitle:
            _narrator_state = "preDebate"
            _ground_statement_text = current_subtitle
            _has_seen_first_debater = False  # Reset this flag when we see ground statement
            _last_detected_speaker = "Narrator"  # Reset speaker tracking
        elif "Summary:" in current_subtitle:
            _ground_statement_summary = current_subtitle.replace("Summary:", "Topic:").strip()
        elif "Result:" in current_subtitle:
            _narrator_state = "postDebate"
            _last_detected_speaker = "Narrator"  # Reset speaker tracking
        # Check for AI Debater and update state
        elif ("AI Debater" in current_subtitle or speaker in ["Jane", "Valentino"]) and _narrator_state == "preDebate":
            # When first debater starts speaking, transition to debate state
            _has_seen_first_debater = True
            _narrator_state = "debate"
            
            # Make sure we have the summary when transitioning to debate state
            if not _ground_statement_summary:
                _ground_statement_summary = "Topic: " + get_ground_statement_summary()
    
    # Improve speaker detection with stability
    determined_speaker = None
    
    # First, use the segment's assigned speaker if available
    if speaker in ["Jane", "Valentino", "Narrator"]:
        determined_speaker = speaker
    
    # If we have timing-based detection
    if current_speaker:
        # If we have both and they conflict, prefer segment speaker for Jane/Valentino
        if determined_speaker and determined_speaker != current_speaker:
            if determined_speaker in ["Jane", "Valentino"]:
                # Keep the segment speaker for debaters
                pass
            else:
                # Use timing speaker
                determined_speaker = current_speaker
        else:
            determined_speaker = current_speaker
    
    # Apply speaker stability - maintain the same speaker during continuous speech
    if current_subtitle:  # Only update when there's active speech
        if determined_speaker:
            if _last_detected_speaker != determined_speaker:
                # Speaker appears to have changed, start counter
                _speaker_stability_counter = 1
                _last_detected_speaker = determined_speaker
            else:
                # Same speaker continues, increase stability
                _speaker_stability_counter += 1
    else:
        # No active speech, gradually decrease stability
        _speaker_stability_counter = max(0, _speaker_stability_counter - 1)
        
        # Reset speaker after period of silence
        if _speaker_stability_counter == 0:
            _last_detected_speaker = None
    
    # Get the final active speaker, with preference for stable detection
    active_speaker = _last_detected_speaker if _speaker_stability_counter > 0 else None
    
    # For avatar highlighting, only use active speaker if it's a debater
    is_debater_speaking = active_speaker in ["Jane", "Valentino"]
    
    # For all cases, show the avatars but only highlight the active debater
    if _jane_avatar and _valentino_avatar:
        # Reset highlight status first - no highlight by default
        _jane_avatar.set_highlight(False)
        _valentino_avatar.set_highlight(False)
        
        # Only highlight if we have an active debater and there's actual content being spoken
        if is_debater_speaking and current_subtitle:
            # Set highlight based on who is currently speaking
            if active_speaker == "Jane":
                _jane_avatar.set_highlight(True)
            elif active_speaker == "Valentino":
                _valentino_avatar.set_highlight(True)
        
        # Draw avatars on frame
        _jane_avatar.draw_on_frame(frame)
        _valentino_avatar.draw_on_frame(frame)

    # Create a PIL Image for text rendering - convert from BGR to RGB for PIL
    pil_img = None
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Draw names under avatars
        if _jane_avatar and _valentino_avatar:
            _jane_avatar.draw_name(draw)
            _valentino_avatar.draw_name(draw)
        
        # TOP TEXT CONTAINER - For ground statement summary during debate or narrator text otherwise
        _top_text.clear()  # Clear previous content
        
        # During debate, always show the summary if available
        if _narrator_state == "debate":
            # If summary is not set, try to get it from the file
            if not _ground_statement_summary:
                _ground_statement_summary = "Topic: " + get_ground_statement_summary()
                print(f"Loaded summary from file for display: {_ground_statement_summary}")
            
            if _ground_statement_summary:
                _top_text.update_text(_ground_statement_summary)
        # During pre debate, show the narrator's text, including introduction and ground statement
        elif _narrator_state == "preDebate":
            if speaker == "Narrator" and current_subtitle:
                # Skip "Summary:" as it's not meant to be spoken
                if not "Summary:" in current_subtitle:
                    # Show whatever the narrator is currently saying during preDebate
                    _top_text.update_text(current_subtitle)
        # During post debate, show the result
        elif _narrator_state == "postDebate" and current_subtitle and "Result:" in current_subtitle:
            _top_text.update_text(current_subtitle)
        
        # Draw the top text
        _top_text.draw(draw)
        
        # BOTTOM TEXT CONTAINER - For debater subtitles
        _bottom_text.clear()  # Clear previous content
        
        if current_subtitle and speaker in ["Jane", "Valentino"]:
            _bottom_text.update_text(current_subtitle)
            _bottom_text.draw(draw)
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    finally:
        if isinstance(pil_img, Image.Image):
            try:
                pil_img.close()
            except:
                pass

def create_frame_worker(args):
    """Worker function for parallel frame creation"""
    try:
        speaker, text, highlighted, current_time, total_duration, timing_segments = args
        
        # Get the current subtitle and speaker from timing data
        current_subtitle, current_speaker = get_current_subtitle(timing_segments, current_time, text)
        
        # Only highlight if:
        # 1. There's subtitle content
        # 2. We're not in narrator mode
        # 3. The actual speaker is determined from timing data
        should_highlight = False
        if current_subtitle and current_speaker:
            if current_speaker in ["Jane", "Valentino"] and speaker != "Narrator":
                should_highlight = (current_speaker == speaker)
        
        frame = create_frame(speaker, text, should_highlight, current_time, total_duration, timing_segments)
        
        # Verify frame is valid
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        # Ensure frame has correct dimensions
        if frame.shape != (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
            return None
        return frame
    except Exception as e:
        print(f"Error in frame worker: {str(e)}")
        return None

def validate_clip_audio(clip, index):
    """
    Validate and fix audio in clip using the VideoClip class.
    
    Args:
        clip: MoviePy VideoClip object
        index: Clip index for logging
        
    Returns:
        MoviePy VideoClip object with validated audio
    """
    # Wrap the clip in our VideoClip class
    video_clip = VideoClip(clip, index)
    # Use the validate_audio method and return the raw clip
    return video_clip.validate_audio().get_raw_clip()

def fix_video_duration(clip, index):
    """
    Function maintained for compatibility - no longer applies safety margin.
    
    Args:
        clip: MoviePy VideoClip object
        index: Clip index for logging
    
    Returns:
        The same clip without modification
    """
    # Simply return the clip without modification
    return clip

def create_segment_video(segment_index, speaker, text, audio_file):
    """Creates a video clip for a single segment."""
    duration = max(get_segment_duration(audio_file), 3.0)
    timing_segments = get_segment_timing(audio_file)
    
    # Generate frames with evenly distributed timestamps
    frames = []
    frames_per_second = 2  # Generate frames at 2fps for smooth subtitles
    total_frames = max(int(duration * frames_per_second), 1)
    time_step = duration / total_frames
    
    for j in range(total_frames):
        time_point = j * time_step
        frame = create_frame(speaker, text, True, time_point, duration, timing_segments)
        if frame is not None:
            frames.append(frame)
    
    # Create fallback frame if necessary
    if not frames:
        print(f"Warning: No frames created for segment {segment_index}. Creating fallback frame.")
        frame = create_frame(speaker, text, True, 0, duration, None)
        if frame is not None:
            frames.append(frame)
    
    try:
        clip = ImageSequenceClip(frames, fps=frames_per_second)
        clip = clip.set_duration(duration)
        # Add audio
        audio_clip = AudioFileClip(audio_file)
        clip = clip.set_audio(audio_clip)
        return clip
    except Exception as e:
        print(f"Error creating clip for segment {segment_index}: {str(e)}")
        # Fallback to simple clip
        try:
            print(f"Using fallback clip for segment {segment_index}")
            fallback_frame = create_frame(speaker, text, True, 0, duration, None)
            if fallback_frame is not None:
                clip = ImageSequenceClip([fallback_frame], fps=1)
                clip = clip.set_duration(duration)
                audio_clip = AudioFileClip(audio_file)
                clip = clip.set_audio(audio_clip)
                return clip
        except Exception as e:
            print(f"Skipping segment {segment_index} due to errors")
        return None

def write_temp_video(clip, index, num_cores):
    """Writes a single clip to a temporary file."""
    temp_file = os.path.join(TEMP_FRAMES_DIR, f"seg_{index:03d}.mp4")
        
    try:
        # Validate clip audio before writing
        clip = validate_clip_audio(clip, index)
        # Fix video duration to prevent frame reading issues
        clip = fix_video_duration(clip, index)
        # Write the clip to a temporary file
        clip.write_videofile(
            temp_file,
            fps=FPS,
            codec='libx264',
            audio_codec='aac',
            preset='ultrafast',
            threads=num_cores,
            verbose=False,
            logger=None,
            temp_audiofile=os.path.join(PROJECT_TEMP_DIR, f"temp_audio_{index}.m4a"),
            ffmpeg_params=["-avoid_negative_ts", "1"]
        )
        return temp_file
    except Exception as e:
        print(f"Error writing segment {index}: {str(e)}")
        return None
    finally:
        # Ensure clip is closed even if there's an error
        try:
            clip.close()
        except:
            pass

def combine_video_segments(clips, output_file):
    """Combines multiple video clips into a final video."""
    # Get number of cores for processing
    num_cores = max(mp.cpu_count() - 1, 1)
    video_clips = []
    temp_files = []
    
    try:
        print("\nCombining video segments...")
        # Write clips one by one to temp files, then concatenate them
        for i, clip in enumerate(clips):
            temp_file = write_temp_video(clip, i, num_cores)
            if temp_file:
                temp_files.append(temp_file)
        clips.clear()
        gc.collect()
        
        print("Creating final video...")
        # Read back all temp videos
        for i, temp_file in enumerate(temp_files):
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                try:
                    clip = VideoFileClip(temp_file, target_resolution=None, resize_algorithm='fast_bilinear')
                    clip = validate_clip_audio(clip, i)
                    video_clips.append(clip)
                except Exception as e:
                    print(f"Error loading clip {temp_file}: {str(e)}")
        
        if video_clips:
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
                return True
            except Exception as e:
                print(f"Error with chain concatenation: {str(e)}")
                # Try alternative concatenation methods
                return _try_alternative_concatenation(video_clips, temp_files, output_file, num_cores)
        else:
            print("No valid video clips were found.")
            return False
    finally:
        # Ensure all video clips are closed before attempting to delete temp files
        for clip in video_clips:
            try:
                clip.close()
            except:
                pass

def _try_alternative_concatenation(video_clips, temp_files, output_file, num_cores):
    """Try alternative concatenation methods if the primary method fails."""
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
                    clip = fix_video_duration(clip, i)
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
            return True
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
                return True
        except Exception as e:
            print(f"All methods failed: {str(e)}")
        return False

def fast_concatenate_clips(clips, method='compose'):
    """
    Faster concatenation of video clips with optimized settings.
    
    Args:
        clips: List of MoviePy VideoClip objects to concatenate
        method: Method for concatenation ('compose' is faster than 'chain')
    
    Returns:
        Concatenated MoviePy VideoClip
    """
    # Wrap each clip in a VideoClip object
    wrapped_clips = [VideoClip(clip, i) for i, clip in enumerate(clips)]
    
    # Use the static concatenate method
    result_clip = VideoClip.concatenate(wrapped_clips, method)
    
    # Return the underlying MoviePy clip
    return result_clip.get_raw_clip() if result_clip else None

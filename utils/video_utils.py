import cv2
import numpy as np
from PIL import Image, ImageDraw
import gc

from config import VIDEO_WIDTH, VIDEO_HEIGHT, TEXT_COLOR, HIGHLIGHT_COLOR
from config import JANE_AVATAR, VALENTINO_AVATAR, TEXT_FONT, NAME_FONT
from utils.text_utils import wrap_text, get_font_metrics
from utils.audio_utils import get_current_subtitle

def create_frame(speaker, text, highlighted=False, current_time=0, total_duration=5.0, timing_segments=None):
    """Create a video frame with speakers and text."""
    # Create blank frame
    frame = np.ones((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8) * 240
    
    # Get current subtitle text and the actual speaker from timing
    current_subtitle, current_speaker = get_current_subtitle(timing_segments, current_time, text)
    
    # Use the detected speaker from timing if available, otherwise use the provided speaker
    active_speaker = current_speaker if current_speaker else speaker
    
    # Only highlight if there's actual subtitle text to display AND
    # the current speaker is one of the debaters (not the narrator) AND
    # we're in a debater section of the video
    should_highlight = bool(current_subtitle) and active_speaker in ["Jane", "Valentino"]
    
    # For narrator introduction, only show text without avatars
    if speaker == "Narrator":
        pil_img = None
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            if current_subtitle:
                # Wrap text to fit screen width with padding
                max_width = VIDEO_WIDTH - 100  # Leave 50px padding on each side
                wrapped_lines = wrap_text(current_subtitle, TEXT_FONT, max_width)
                
                # Calculate total height of text block
                _, line_height = get_font_metrics(TEXT_FONT, "Tg")
                line_spacing = 10
                total_height = len(wrapped_lines) * (line_height + line_spacing)
                
                # Start position for first line (centered vertically)
                text_y = (VIDEO_HEIGHT - total_height) // 2
                
                # Draw each line centered horizontally - without background
                for line in wrapped_lines:
                    text_width, _ = get_font_metrics(TEXT_FONT, line)
                    text_x = (VIDEO_WIDTH - text_width) // 2
                    
                    # Draw text directly without background rectangle or shadow
                    draw.text((text_x, text_y), line, fill=TEXT_COLOR, font=TEXT_FONT)
                    
                    text_y += line_height + line_spacing
            
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        finally:
            if isinstance(pil_img, Image.Image):
                try:
                    pil_img.close()
                except:
                    pass

    # For debaters, show avatars and only highlight the current speaker
    if JANE_AVATAR is not None and VALENTINO_AVATAR is not None:
        # Move avatars up by 50 pixels
        avatar_vertical_offset = 50
        jane_pos = (100, (VIDEO_HEIGHT - 250 - 50) // 2 - avatar_vertical_offset)
        valentino_pos = (VIDEO_WIDTH - 250 - 100, (VIDEO_HEIGHT - 250 - 50) // 2 - avatar_vertical_offset)
        
        # Draw avatars on frame
        frame[jane_pos[1]:jane_pos[1]+250, jane_pos[0]:jane_pos[0]+250] = JANE_AVATAR
        frame[valentino_pos[1]:valentino_pos[1]+250, valentino_pos[0]:valentino_pos[0]+250] = VALENTINO_AVATAR
        
        # Add highlight effect ONLY when a debater is actively speaking
        # Never highlight during narrator sections
        if should_highlight:
            highlight_thickness = 10
            if active_speaker == "Jane":
                cv2.rectangle(
                    frame,
                    (jane_pos[0]-highlight_thickness, jane_pos[1]-highlight_thickness),
                    (jane_pos[0]+250+highlight_thickness, jane_pos[1]+250+highlight_thickness),
                    HIGHLIGHT_COLOR,
                    highlight_thickness
                )
            elif active_speaker == "Valentino":
                cv2.rectangle(
                    frame,
                    (valentino_pos[0]-highlight_thickness, valentino_pos[1]-highlight_thickness),
                    (valentino_pos[0]+250+highlight_thickness, valentino_pos[1]+250+highlight_thickness),
                    HIGHLIGHT_COLOR,
                    highlight_thickness
                )

    # Create a PIL Image for text rendering
    pil_img = None
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Add names under avatars if not narrator
        if speaker != "Narrator":
            jane_name_y = jane_pos[1] + 250 + 10
            valentino_name_y = valentino_pos[1] + 250 + 10
            
            jane_name_width, _ = get_font_metrics(NAME_FONT, "Jane")
            valentino_name_width, _ = get_font_metrics(NAME_FONT, "Valentino")
            
            jane_name_x = jane_pos[0] + (250 - jane_name_width) // 2
            valentino_name_x = valentino_pos[0] + (250 - valentino_name_width) // 2
            
            draw.text((jane_name_x, jane_name_y), "Jane", fill=TEXT_COLOR, font=NAME_FONT)
            draw.text((valentino_name_x, valentino_name_y), "Valentino", fill=TEXT_COLOR, font=NAME_FONT)
        
        # For subtitle text - show only if we have subtitle content
        if current_subtitle:
            # Add speaker indicator to the subtitle if in a debate section
            subtitle_text = current_subtitle
            if active_speaker in ["Jane", "Valentino"] and speaker != "Narrator":
                subtitle_text = f"{active_speaker}: {current_subtitle}"
            
            # Wrap subtitle text
            max_width = VIDEO_WIDTH - 100  # Leave 50px padding on each side
            wrapped_lines = wrap_text(subtitle_text, TEXT_FONT, max_width)
            
            # Calculate total height of text block
            _, line_height = get_font_metrics(TEXT_FONT, "Tg")
            line_spacing = 10
            total_height = len(wrapped_lines) * (line_height + line_spacing)
            
            # Start position for first line - moved lower (closer to bottom)
            subtitle_vertical_offset = 80  # Move subtitles down (higher value = lower position)
            text_y = VIDEO_HEIGHT - subtitle_vertical_offset - total_height
            
            # Draw each line centered horizontally
            for line in wrapped_lines:
                text_width, _ = get_font_metrics(TEXT_FONT, line)
                text_x = (VIDEO_WIDTH - text_width) // 2
                
                # Draw text directly without background or shadow
                draw.text((text_x, text_y), line, fill=TEXT_COLOR, font=TEXT_FONT)
                
                text_y += line_height + line_spacing
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
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
    """Validate and fix audio in clip to prevent buffer boundary issues."""
    try:
        if clip.audio is None:
            print(f"Warning: Clip {index} has no audio, creating silent audio")
            from moviepy.audio.AudioClip import AudioClip
            clip = clip.set_audio(AudioClip(lambda t: 0, duration=clip.duration))
            return clip
        
        # Ensure audio duration matches or is slightly shorter than video
        if abs(clip.audio.duration - clip.duration) > 0.1:
            print(f"Warning: Clip {index} audio/video duration mismatch: Audio={clip.audio.duration:.2f}s, Video={clip.duration:.2f}s")
            
            # Instead of trimming audio, extend the video duration to fit the audio
            if clip.audio.duration > clip.duration:
                print(f"Extending video duration to match audio for clip {index}")
                # Create a copy of the clip with the last frame extended
                from moviepy.video.VideoClip import ImageClip
                last_frame = clip.get_frame(clip.duration - 0.01)  # Get the last frame
                extension = ImageClip(last_frame).set_duration(clip.audio.duration - clip.duration)
                from moviepy.video.compositing.concatenate import concatenate_videoclips
                extended_clip = concatenate_videoclips([clip, extension])
                extended_clip = extended_clip.set_audio(clip.audio)
                print(f"Extended video from {clip.duration:.2f}s to {extended_clip.duration:.2f}s")
                return extended_clip
            else:
                # If audio is too short, extend with silence (keep existing behavior)
                from moviepy.audio.AudioClip import AudioClip, concatenate_audioclips
                silence = AudioClip(lambda t: 0, duration=clip.duration - clip.audio.duration)
                clip = clip.set_audio(concatenate_audioclips([clip.audio, silence]))
                print(f"Extended audio with silence for clip {index}")
        
        return clip
    except Exception as e:
        print(f"Error validating clip {index} audio: {str(e)}")
        return clip

def fix_video_duration(clip, index):
    """Fix video duration to prevent frame reading issues at the end."""
    try:
        # Only apply a minimal safety margin if needed
        if clip.duration > 1.0:  # Only for clips longer than 1 second
            # Reduce the safety margin to minimize content loss
            safety_margin = 0.01  # 10ms safety margin instead of 50ms
            new_duration = clip.duration - safety_margin
            print(f"Applied minimal safety margin to clip {index}: {clip.duration:.2f}s → {new_duration:.2f}s")
            return clip.subclip(0, new_duration)
        return clip
    except Exception as e:
        print(f"Error fixing clip {index} duration: {str(e)}")
        return clip

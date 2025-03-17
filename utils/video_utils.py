import cv2
import numpy as np
from PIL import Image, ImageDraw
import gc

from config import VIDEO_WIDTH, VIDEO_HEIGHT, TEXT_COLOR, HIGHLIGHT_COLOR
from config import JANE_AVATAR, VALENTINO_AVATAR, TEXT_FONT, NAME_FONT
from utils.text_utils import wrap_text, get_font_metrics
from utils.audio_utils import get_current_subtitle

# Track debate state and text content
_narrator_state = "preDebate"  # "preDebate", "debate", "postDebate"
_ground_statement_text = ""
_ground_statement_summary = ""
_has_seen_first_debater = False  # Flag to track if we've seen the first debater

def create_frame(speaker, text, highlighted=False, current_time=0, total_duration=5.0, timing_segments=None):
    """Create a video frame with speakers and text."""
    global _narrator_state, _ground_statement_text, _ground_statement_summary, _has_seen_first_debater
    
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
        elif "Display Summary:" in current_subtitle:
            _ground_statement_summary = current_subtitle.replace("Display Summary:", "Topic:").strip()
        elif "Result:" in current_subtitle:
            _narrator_state = "postDebate"
        # Check for AI Debater and update state
        elif ("AI Debater" in current_subtitle or speaker in ["Jane", "Valentino"]) and _narrator_state == "preDebate":
            # When first debater starts speaking, transition to debate state
            _has_seen_first_debater = True
            _narrator_state = "debate"
    
    # Use the detected speaker from timing if available, otherwise use the provided speaker
    active_speaker = current_speaker if current_speaker else speaker
    
    # Only highlight if there's actual subtitle text to display AND speaker is a debater
    should_highlight = bool(current_subtitle) and active_speaker in ["Jane", "Valentino"]
    
    # For all cases (including narrator), show the avatars but only highlight active debater
    if JANE_AVATAR is not None and VALENTINO_AVATAR is not None:
        # Move avatars up by 50 pixels
        avatar_vertical_offset = 50
        jane_pos = (100, (VIDEO_HEIGHT - 250 - 50) // 2 - avatar_vertical_offset)
        valentino_pos = (VIDEO_WIDTH - 250 - 100, (VIDEO_HEIGHT - 250 - 50) // 2 - avatar_vertical_offset)
        
        # Draw avatars on frame - OpenCV uses BGR format
        frame[jane_pos[1]:jane_pos[1]+250, jane_pos[0]:jane_pos[0]+250] = JANE_AVATAR
        frame[valentino_pos[1]:valentino_pos[1]+250, valentino_pos[0]:valentino_pos[0]+250] = VALENTINO_AVATAR
        
        # Add highlight effect ONLY when a debater is actively speaking
        # Never highlight during narrator sections
        if should_highlight and speaker != "Narrator":
            highlight_thickness = 10
            # Fix the highlight color - HIGHLIGHT_COLOR is likely defined as RGB but OpenCV expects BGR
            highlight_color_bgr = (HIGHLIGHT_COLOR[2], HIGHLIGHT_COLOR[1], HIGHLIGHT_COLOR[0])
            
            if active_speaker == "Jane":
                cv2.rectangle(
                    frame,
                    (jane_pos[0]-highlight_thickness, jane_pos[1]-highlight_thickness),
                    (jane_pos[0]+250+highlight_thickness, jane_pos[1]+250+highlight_thickness),
                    highlight_color_bgr,  # Use BGR order for OpenCV
                    highlight_thickness
                )
            elif active_speaker == "Valentino":
                cv2.rectangle(
                    frame,
                    (valentino_pos[0]-highlight_thickness, valentino_pos[1]-highlight_thickness),
                    (valentino_pos[0]+250+highlight_thickness, valentino_pos[1]+250+highlight_thickness),
                    highlight_color_bgr,  # Use BGR order for OpenCV
                    highlight_thickness
                )

    # Create a PIL Image for text rendering - convert from BGR to RGB for PIL
    pil_img = None
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Add names under avatars
        jane_name_y = jane_pos[1] + 250 + 10
        valentino_name_y = valentino_pos[1] + 250 + 10
        
        jane_name_width, _ = get_font_metrics(NAME_FONT, "Jane")
        valentino_name_width, _ = get_font_metrics(NAME_FONT, "Valentino")
        
        jane_name_x = jane_pos[0] + (250 - jane_name_width) // 2
        valentino_name_x = valentino_pos[0] + (250 - valentino_name_width) // 2
        
        draw.text((jane_name_x, jane_name_y), "Jane", fill=TEXT_COLOR, font=NAME_FONT)
        draw.text((valentino_name_x, valentino_name_y), "Valentino", fill=TEXT_COLOR, font=NAME_FONT)
        
        # TOP TEXT CONTAINER - For ground statement summary during debate or narrator text otherwise
        top_text = None
        
        # During debate, always show the summary if available
        if _narrator_state == "debate" and _ground_statement_summary:
            top_text = _ground_statement_summary
        # During pre debate, show the narrator's text, including introduction and ground statement
        elif _narrator_state == "preDebate":
            if speaker == "Narrator" and current_subtitle:
                # Skip "Display Summary:" as it's not meant to be spoken
                if not "Display Summary:" in current_subtitle:
                    # Show whatever the narrator is currently saying during preDebate
                    top_text = current_subtitle
            elif _ground_statement_text and not speaker == "Narrator":
                # For non-narrator speakers in preDebate, show the full ground statement
                top_text = _ground_statement_text
        # During post debate, show the result
        elif _narrator_state == "postDebate" and current_subtitle and "Result:" in current_subtitle:
            top_text = current_subtitle
        
        # Draw top text if we have content
        if top_text:
            max_width = VIDEO_WIDTH - 300
            wrapped_lines = wrap_text(top_text, TEXT_FONT, max_width)
            
            # Calculate total height of text block
            _, line_height = get_font_metrics(TEXT_FONT, "Tg")
            line_spacing = 10
            total_height = len(wrapped_lines) * (line_height + line_spacing)
            
            # Position text at the top with padding
            top_padding = 20
            text_y = top_padding
            
            # Draw each line centered horizontally
            for line in wrapped_lines:
                text_width, _ = get_font_metrics(TEXT_FONT, line)
                text_x = (VIDEO_WIDTH - text_width) // 2
                
                # Add background for text at the top (always)
                text_bg_padding = 10
                text_bg = (220, 220, 220, 180)  # Light gray with some transparency
                
                # Draw rounded rectangle background
                draw.rectangle(
                    [(text_x - text_bg_padding, text_y - text_bg_padding/2),
                     (text_x + text_width + text_bg_padding, text_y + line_height + text_bg_padding/2)],
                    fill=text_bg
                )
                
                # Draw the text
                draw.text((text_x, text_y), line, fill=TEXT_COLOR, font=TEXT_FONT)
                text_y += line_height + line_spacing
        
        # BOTTOM TEXT CONTAINER - For debater subtitles
        if current_subtitle and speaker in ["Jane", "Valentino"]:
            # For debater subtitles - always position at the bottom
            max_width = VIDEO_WIDTH - 100  # Leave 50px padding on each side
            wrapped_lines = wrap_text(current_subtitle, TEXT_FONT, max_width)
            
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

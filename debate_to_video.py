import os
import cv2
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips  # Fixed import path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
import re
from datetime import datetime
import subprocess

# Constants for video generation
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 24
BACKGROUND_COLOR = (240, 240, 240)  # Light gray background
HIGHLIGHT_COLOR = (255, 223, 0)  # Yellow highlight
TEXT_COLOR = (0, 0, 0)  # Black text
GROUND_STATEMENT_COLOR = (40, 40, 40)  # Dark gray for ground statement

def parse_debate_file():
    """Parse debate.txt file to get dialogue segments and speakers."""
    dialogue_segments = []
    
    try:
        with open('outputs/debate.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("Ground Statement:"):
                    speaker = "Ground"
                    text = line  # Keep the full line for ground statement
                elif line.startswith("AI Debater 1:"):
                    speaker = "Jane"
                    text = line.replace("AI Debater 1:", "").strip()
                elif line.startswith("AI Debater 2:"):
                    speaker = "Valentino"
                    text = line.replace("AI Debater 2:", "").strip()
                elif line.startswith("Result:"):
                    speaker = "Result"
                    text = line  # Keep the full line for result
                else:
                    continue
                    
                dialogue_segments.append({"speaker": speaker, "text": text})
    except (FileNotFoundError, IOError) as e:
        print(f"Error reading debate file: {str(e)}")
    
    return dialogue_segments

def get_segment_audio_file(index):
    """Get the audio file for a specific segment."""
    # Look for audio files in the audio_output folder
    files = os.listdir('outputs/audio_output')
    # Sort to ensure we get them in the correct order
    files.sort()
    
    # Match the index to the file
    if index < len(files):
        return os.path.join('outputs/audio_output', files[index])
    else:
        print(f"Warning: No audio file found for segment {index}")
        return None

def get_segment_duration(audio_file):
    """Get the duration of an audio segment."""
    if not audio_file or not os.path.exists(audio_file):
        return 5.0  # Default duration if no audio file
        
    try:
        audio = AudioSegment.from_mp3(audio_file)
        return len(audio) / 1000.0  # Convert milliseconds to seconds
    except Exception as e:
        print(f"Error getting audio duration: {str(e)}")
        return 5.0  # Default duration on error

def create_frame(speaker, text, highlighted=False):
    """Create a video frame with speakers and text."""
    # Create blank frame
    frame = np.ones((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Load speaker avatars
    try:
        jane_img = cv2.imread('assets/jane_avatar.jpg')
        valentino_img = cv2.imread('assets/valentino_avatar.jpg')
        
        if jane_img is not None and valentino_img is not None:
            # Resize avatars to appropriate size
            avatar_size = 250  # Slightly smaller to make room for names
            jane_img = cv2.resize(jane_img, (avatar_size, avatar_size))
            valentino_img = cv2.resize(valentino_img, (avatar_size, avatar_size))
            
            # Calculate positions for avatars
            jane_pos = (100, (VIDEO_HEIGHT - avatar_size - 50) // 2)  # Move up to make room for name
            valentino_pos = (VIDEO_WIDTH - avatar_size - 100, (VIDEO_HEIGHT - avatar_size - 50) // 2)
            
            # Draw avatars on frame
            frame[jane_pos[1]:jane_pos[1]+avatar_size, jane_pos[0]:jane_pos[0]+avatar_size] = jane_img
            frame[valentino_pos[1]:valentino_pos[1]+avatar_size, valentino_pos[0]:valentino_pos[0]+avatar_size] = valentino_img
            
            # Add highlight effect when speaker is active
            if highlighted:
                highlight_thickness = 10
                if speaker == "Jane":
                    cv2.rectangle(
                        frame,
                        (jane_pos[0]-highlight_thickness, jane_pos[1]-highlight_thickness),
                        (jane_pos[0]+avatar_size+highlight_thickness, jane_pos[1]+avatar_size+highlight_thickness),
                        HIGHLIGHT_COLOR,
                        highlight_thickness
                    )
                elif speaker == "Valentino":
                    cv2.rectangle(
                        frame,
                        (valentino_pos[0]-highlight_thickness, valentino_pos[1]-highlight_thickness),
                        (valentino_pos[0]+avatar_size+highlight_thickness, valentino_pos[1]+avatar_size+highlight_thickness),
                        HIGHLIGHT_COLOR,
                        highlight_thickness
                    )
    except Exception as e:
        print(f"Error loading avatar images: {str(e)}")
    
    # Create a PIL Image for text rendering
    pil_img = None
    frame_bgr = None
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Try to load fonts with different sizes
        try:
            name_font_size = 36
            text_font_size = 32
            name_font = ImageFont.truetype("arial.ttf", name_font_size)
            text_font = ImageFont.truetype("arial.ttf", text_font_size)
        except:
            name_font = text_font = ImageFont.load_default()
        
        # Add names under avatars
        jane_name_y = jane_pos[1] + avatar_size + 10
        valentino_name_y = valentino_pos[1] + avatar_size + 10
        
        # Calculate name positions to center them under avatars
        jane_name = "Jane"
        valentino_name = "Valentino"
        
        try:
            jane_name_width = name_font.getlength(jane_name)
        except:
            jane_name_width = name_font.getsize(jane_name)[0]
        try:
            valentino_name_width = name_font.getlength(valentino_name)
        except:
            valentino_name_width = name_font.getsize(valentino_name)[0]
        
        jane_name_x = jane_pos[0] + (avatar_size - jane_name_width) // 2
        valentino_name_x = valentino_pos[0] + (avatar_size - valentino_name_width) // 2
        
        # Draw names
        draw.text((jane_name_x, jane_name_y), jane_name, fill=TEXT_COLOR, font=name_font)
        draw.text((valentino_name_x, valentino_name_y), valentino_name, fill=TEXT_COLOR, font=name_font)
        
        # For subtitle text, show only a short part
        if speaker in ["Ground", "Result"]:
            # Keep the prefix for ground statement and result
            subtitle_text = text
            color = GROUND_STATEMENT_COLOR
        else:
            # For regular dialogue, limit the text length and remove speaker prefix
            words = text.split()
            if len(words) > 12:  # Show around 12 words at a time
                subtitle_text = " ".join(words[:12]) + "..."
            else:
                subtitle_text = text
            color = TEXT_COLOR
        
        # Calculate position for subtitle (centered at bottom)
        try:
            text_width = text_font.getlength(subtitle_text)
        except:
            text_width = text_font.getsize(subtitle_text)[0]
            
        text_x = (VIDEO_WIDTH - text_width) // 2
        text_y = VIDEO_HEIGHT - 100  # Position near bottom
        
        # Draw subtitle text with a semi-transparent background for better readability
        padding = 10
        text_bbox = draw.textbbox((text_x, text_y), subtitle_text, font=text_font)
        draw.rectangle((text_bbox[0]-padding, text_bbox[1]-padding, 
                       text_bbox[2]+padding, text_bbox[3]+padding), 
                      fill=(240, 240, 240, 230))  # Light gray background
        draw.text((text_x, text_y), subtitle_text, fill=color, font=text_font)
        
        # Convert back to OpenCV format
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return frame_bgr
    except Exception as e:
        print(f"Error creating frame: {str(e)}")
        return frame
    finally:
        # Properly handle PIL Image cleanup
        if isinstance(pil_img, Image.Image):
            try:
                pil_img.close()
            except:
                pass

def wrap_text(text, font, max_width):
    """Wrap text to fit within max_width."""
    words = text.split()
    wrapped_lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + word + " "
        # Use getsize for older PIL versions or getbbox/getlength for newer versions
        try:
            text_width = font.getlength(test_line)
        except AttributeError:
            try:
                text_width = font.getbbox(test_line)[2]  # width from bbox
            except:
                text_width = font.getsize(test_line)[0]  # fallback to getsize
                
        if text_width <= max_width:
            current_line = test_line
        else:
            wrapped_lines.append(current_line)
            current_line = word + " "
            
    wrapped_lines.append(current_line)
    return wrapped_lines

def create_debate_video():
    """Create a video visualization of the debate with audio."""
    print("Generating debate video...")
    
    # Parse debate file
    dialogue_segments = parse_debate_file()
    if not dialogue_segments:
        print("No dialogue segments found. Check debate.txt file.")
        return
    
    # Create temporary directory for frames
    os.makedirs("outputs/temp_frames", exist_ok=True)
    
    # Create video clips for each segment
    clip_segments = []
    audio_clips = []  # Track audio clips for cleanup
    
    try:
        for i, segment in enumerate(dialogue_segments):
            speaker = segment["speaker"]
            text = segment["text"]
            
            # Get audio file and duration for this segment
            audio_file = get_segment_audio_file(i)
            duration = get_segment_duration(audio_file)
            
            # Create frame for this segment
            frame = create_frame(speaker, text, highlighted=True)
            
            # Save frame as image
            frame_path = f"outputs/temp_frames/frame_{i:03d}.png"
            success = cv2.imwrite(frame_path, frame)
            if not success:
                print(f"Error writing frame {i} to {frame_path}")
                continue
            
            # Create video clip from the frame
            clip = ImageClip(frame_path).set_duration(duration)
            
            # If audio file exists, add it to the clip
            if audio_file and os.path.exists(audio_file):
                audio_clip = AudioFileClip(audio_file)
                audio_clips.append(audio_clip)  # Track for cleanup
                clip = clip.set_audio(audio_clip)
            
            clip_segments.append(clip)
        
        # Concatenate all clips
        if clip_segments:
            final_clip = concatenate_videoclips(clip_segments)
            
            # Create timestamp for output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"outputs/debate_{timestamp}.mp4"
            
            try:
                # Write the final video file
                final_clip.write_videofile(output_file, fps=FPS, codec='libx264', audio_codec='aac')
                print(f"Debate video created: {output_file}")
            finally:
                # Clean up the final clip
                final_clip.close()
        else:
            print("No video clips were generated.")
    
    finally:
        # Clean up all clips
        for clip in clip_segments:
            try:
                clip.close()
            except:
                pass
        
        # Clean up all audio clips
        for clip in audio_clips:
            try:
                clip.close()
            except:
                pass
        
        # Clean up temporary files
        try:
            for file in os.listdir("outputs/temp_frames"):
                os.remove(os.path.join("outputs/temp_frames", file))
            os.rmdir("outputs/temp_frames")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    create_debate_video()
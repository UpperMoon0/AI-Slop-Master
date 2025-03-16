import os
import cv2
from moviepy.editor import AudioFileClip, ImageSequenceClip, concatenate_videoclips, VideoFileClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
import json
import multiprocessing as mp
from functools import partial
import gc

# Constants for video generation
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 24
BACKGROUND_COLOR = (240, 240, 240)
HIGHLIGHT_COLOR = (255, 223, 0)
TEXT_COLOR = (0, 0, 0)
GROUND_STATEMENT_COLOR = (40, 40, 40)

# Cache fonts
try:
    NAME_FONT = ImageFont.truetype("arial.ttf", 36)
    TEXT_FONT = ImageFont.truetype("arial.ttf", 32)
except:
    NAME_FONT = TEXT_FONT = ImageFont.load_default()

# Cache avatar images
try:
    JANE_AVATAR = cv2.resize(cv2.imread('assets/jane_avatar.jpg'), (250, 250))
    VALENTINO_AVATAR = cv2.resize(cv2.imread('assets/valentino_avatar.jpg'), (250, 250))
except:
    JANE_AVATAR = VALENTINO_AVATAR = None

def wrap_text(text, font, max_width):
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        # Try adding the word to the current line
        test_line = current_line + [word]
        test_text = " ".join(test_line)
        text_width, _ = get_font_metrics(font, test_text)
            
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:  # Add the current line if it's not empty
                lines.append(" ".join(current_line))
            current_line = [word]
    
    if current_line:  # Add the last line
        lines.append(" ".join(current_line))
    
    return lines

def parse_debate_file():
    """Parse debate.txt file to get dialogue segments and speakers."""
    dialogue_segments = []
    
    try:
        with open('outputs/debate.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("Narrator:"):
                    speaker = "Narrator"
                    text = line.replace("Narrator:", "").strip()
                elif line.startswith("Ground Statement:"):
                    speaker = "Narrator"  # Changed from "Ground" to "Narrator"
                    text = line  # Keep the full line for ground statement
                elif line.startswith("AI Debater 1:"):
                    speaker = "Jane"
                    text = line.replace("AI Debater 1:", "").strip()
                elif line.startswith("AI Debater 2:"):
                    speaker = "Valentino"
                    text = line.replace("AI Debater 2:", "").strip()
                elif line.startswith("Result:"):
                    speaker = "Narrator"  # Changed from "Result" to "Narrator"
                    text = line  # Keep the full line for result
                else:
                    continue
                    
                dialogue_segments.append({"speaker": speaker, "text": text})
    except (FileNotFoundError, IOError) as e:
        print(f"Error reading debate file: {str(e)}")
    
    return dialogue_segments

def get_segment_audio_file(index):
    """Get the audio file for a specific segment."""
    try:
        files = [f for f in os.listdir('outputs/audio_output') if f.endswith('.mp3')]
        files.sort()
        
        if index < len(files):
            return os.path.join('outputs/audio_output', files[index])
        else:
            print(f"Warning: No audio file found for segment {index}")
            return None
    except Exception:
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

def split_text_into_chunks(text, chunk_size=8):
    """Split text into chunks of roughly equal size."""
    if not text:  # Handle empty text
        return [""]
        
    words = text.split()
    if not words:  # Handle text with no words
        return [""]
        
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    
    if current_chunk:  # Add any remaining words
        chunks.append(" ".join(current_chunk))
    
    return chunks if chunks else [text]  # Return original text if no chunks were created

def get_segment_timing(audio_file):
    """Get the timing information for a specific audio segment."""
    if not audio_file:
        return []
        
    timing_file = audio_file.replace('.mp3', '_timing.json')
    try:
        with open(timing_file, 'r') as f:
            timing_data = json.load(f)
            return timing_data.get("segments", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def get_current_subtitle(timing_segments, current_time, original_text):
    """Get the subtitle text that should be displayed at the current time."""
    if not timing_segments:
        return ""
        
    for segment in timing_segments:
        if segment["start_time"] <= current_time <= segment["end_time"]:
            return original_text  # Use the original text instead of recognized text
            
    return ""  # No subtitle for this time

def get_font_metrics(font, text):
    """Get font metrics in a way that works with both old and new Pillow versions."""
    try:
        # Try new Pillow methods first
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]  # width, height
    except AttributeError:
        try:
            # Try getlength for width (intermediate Pillow versions)
            width = font.getlength(text)
            # For height, we'll use a reference character
            _, height = font.getsize("Tg")
            return width, height
        except AttributeError:
            # Fall back to old getsize method
            return font.getsize(text)

def create_frame(speaker, text, highlighted=False, current_time=0, total_duration=5.0, timing_segments=None):
    """Create a video frame with speakers and text."""
    # Create blank frame
    frame = np.ones((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8) * 240

    # For narrator introduction, only show text without avatars
    if speaker == "Narrator":
        pil_img = None
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Split text into smaller chunks for display
            chunks = split_text_into_chunks(text, chunk_size=8)
            if chunks:
                chunk_duration = total_duration / len(chunks)
                current_chunk_index = min(int(current_time / chunk_duration), len(chunks) - 1)
                subtitle_text = chunks[current_chunk_index]
                
                # Wrap text to fit screen width with padding
                max_width = VIDEO_WIDTH - 100  # Leave 50px padding on each side
                wrapped_lines = wrap_text(subtitle_text, TEXT_FONT, max_width)
                
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

    # For other speakers, calculate positions with avatars moved higher
    if JANE_AVATAR is not None and VALENTINO_AVATAR is not None:
        # Move avatars up by 50 pixels
        avatar_vertical_offset = 50
        jane_pos = (100, (VIDEO_HEIGHT - 250 - 50) // 2 - avatar_vertical_offset)
        valentino_pos = (VIDEO_WIDTH - 250 - 100, (VIDEO_HEIGHT - 250 - 50) // 2 - avatar_vertical_offset)
        
        # Draw avatars on frame
        frame[jane_pos[1]:jane_pos[1]+250, jane_pos[0]:jane_pos[0]+250] = JANE_AVATAR
        frame[valentino_pos[1]:valentino_pos[1]+250, valentino_pos[0]:valentino_pos[0]+250] = VALENTINO_AVATAR
        
        # Add highlight effect when speaker is active
        if highlighted:
            highlight_thickness = 10
            if speaker == "Jane":
                cv2.rectangle(
                    frame,
                    (jane_pos[0]-highlight_thickness, jane_pos[1]-highlight_thickness),
                    (jane_pos[0]+250+highlight_thickness, jane_pos[1]+250+highlight_thickness),
                    HIGHLIGHT_COLOR,
                    highlight_thickness
                )
            elif speaker == "Valentino":
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
        
        # For subtitle text, use timing information if available
        color = TEXT_COLOR
        subtitle_text = ""
        
        if timing_segments:
            # Find the segment that matches the current time
            for segment in timing_segments:
                if segment["start_time"] <= current_time <= segment["end_time"]:
                    subtitle_text = segment["text"]
                    break
            
            # If no direct match, get the subtitle based on the current position
            if not subtitle_text:
                chunks = split_text_into_chunks(text, chunk_size=8)  # Smaller chunks (8 words)
                if chunks:
                    chunk_duration = total_duration / len(chunks)
                    current_chunk_index = min(int(current_time / chunk_duration), len(chunks) - 1)
                    subtitle_text = chunks[current_chunk_index]
        else:
            chunks = split_text_into_chunks(text, chunk_size=8)  # Smaller chunks (8 words)
            if chunks:
                chunk_duration = total_duration / len(chunks)
                current_chunk_index = min(int(current_time / chunk_duration), len(chunks) - 1)
                subtitle_text = chunks[current_chunk_index]
        
        if subtitle_text:
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
            
            # Draw each line centered horizontally - without background or shadow
            for line in wrapped_lines:
                text_width, _ = get_font_metrics(TEXT_FONT, line)
                text_x = (VIDEO_WIDTH - text_width) // 2
                
                # Draw text directly without background or shadow
                draw.text((text_x, text_y), line, fill=color, font=TEXT_FONT)
                
                text_y += line_height + line_spacing
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    finally:
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

def create_frame_worker(args):
    """Worker function for parallel frame creation"""
    try:
        speaker, text, highlighted, current_time, total_duration, timing_segments = args
        frame = create_frame(speaker, text, highlighted, current_time, total_duration, timing_segments)
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

def create_debate_video():
    """Create a video visualization of the debate with audio."""
    dialogue_segments = parse_debate_file()
    if not dialogue_segments:
        print("No dialogue segments found. Aborting video creation.")
        return
    
    # Initialize multiprocessing pool with fewer cores to reduce memory pressure
    num_cores = max(min(mp.cpu_count() - 1, 2), 1)  # Use at most 2 cores
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
                print(f"Generated clip for segment {i+1}/{len(dialogue_segments)}")
                
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
                    temp_file = f"outputs/temp_frames/seg_{i:03d}.mp4"
                    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
                    
                    print(f"Writing segment {i+1}/{len(all_clips)}...")
                    clip.write_videofile(
                        temp_file,
                        fps=24,
                        codec='libx264',
                        audio_codec='aac',
                        preset='ultrafast',  # Use fastest preset to reduce processing time
                        threads=num_cores,
                        verbose=False,
                        logger=None
                    )
                    temp_files.append(temp_file)
                    
                    # Close and clean up the clip
                    clip.close()
                
                print("Creating final video...")
                # Read back all temp videos
                video_clips = []
                for temp_file in temp_files:
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        clip = VideoFileClip(temp_file)
                        video_clips.append(clip)
                
                if video_clips:
                    # Concatenate all clips
                    final_clip = concatenate_videoclips(video_clips, method="compose")
                    
                    # Write final video
                    final_clip.write_videofile(
                        output_file,
                        fps=24,
                        codec='libx264',
                        audio_codec='aac',
                        preset='medium',  # Better quality for final output
                        threads=num_cores,
                        verbose=False,
                        logger=None
                    )
                    
                    print(f"Video saved as: {output_file}")
                    
                    # Clean up
                    final_clip.close()
                    for clip in video_clips:
                        clip.close()
                else:
                    print("No valid video clips were found.")
                
                # Clean up temp files
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
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
                            fps=24,
                            codec='libx264',
                            audio_codec='aac',
                            preset='ultrafast',
                            threads=num_cores,
                            verbose=False,
                            logger=None
                        )
                        
                        print(f"Video saved as: {output_file} (simplified version)")
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
        
        print("Cleaning up...")
        # Force garbage collection
        gc.collect()

if __name__ == "__main__":
    # Set process start method for Windows
    if os.name == 'nt':  # Windows
        mp.set_start_method('spawn', force=True)
    create_debate_video()
import asyncio
import os
import shutil
import xml.sax.saxutils
import re
from typing import List, Tuple
import edge_tts
from datetime import datetime
from pydub import AudioSegment
import subprocess

# Using standard Edge TTS voices
VOICE_NARRATOR = "en-US-ChristopherNeural"  # Changed from DavidNeural to more stable ChristopherNeural
VOICE_JANE = "en-US-JennyNeural"
VOICE_VALENTINO = "en-US-GuyNeural"
VOICE_BACKUP = "en-US-EricNeural"  # Backup voice for ground statements

def clear_output_folder():
    """Clear the audio_output folder if it exists."""
    if os.path.exists('outputs/audio_output'):
        shutil.rmtree('outputs/audio_output')
    os.makedirs('outputs/audio_output', exist_ok=True)

def escape_text(text: str) -> str:
    """Escape special characters in text for SSML."""
    return xml.sax.saxutils.escape(text)

def split_long_text(text: str, max_length: int = 1000) -> List[str]:
    """Split long text into smaller chunks at sentence boundaries."""
    if len(text) <= max_length:
        return [text]
    
    # Split at sentence boundaries (periods, question marks, exclamation points)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max_length, start a new chunk
        if len(current_chunk) + len(sentence) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk:  # Add the last chunk if it exists
        chunks.append(current_chunk.strip())
    
    return chunks

async def text_to_speech(text: str, voice: str, output_file: str, is_ground_statement: bool = False) -> None:
    """Convert text to speech using Edge TTS with speed control."""
    max_retries = 3
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Escape special characters in text
    escaped_text = escape_text(text)
    
    # For ground statements, use special handling
    if is_ground_statement:
        # Try splitting the text if it's long
        if len(escaped_text) > 1000:
            chunks = split_long_text(escaped_text)
            temp_files = []
            
            try:
                # Process each chunk separately
                for i, chunk in enumerate(chunks):
                    temp_file = f"{output_file}.part{i}.mp3"
                    temp_files.append(temp_file)
                    
                    # Try with primary voice, fallback to backup voice
                    try:
                        communicate = edge_tts.Communicate(text=chunk, voice=voice, rate="-10%")  # Changed from "0%" to "-10%"
                        await communicate.save(temp_file)
                        print(f"Generated chunk {i+1}/{len(chunks)} of ground statement")
                    except Exception:
                        print(f"Retrying chunk {i+1} with backup voice")
                        communicate = edge_tts.Communicate(text=chunk, voice=VOICE_BACKUP, rate="-10%")  # Changed from "0%" to "-10%"
                        await communicate.save(temp_file)
                    
                    # Reduced sleep time between chunks
                    await asyncio.sleep(0.2)
                
                # Combine the temporary files
                if temp_files:
                    # Use pydub to combine the chunks
                    combined = AudioSegment.empty()
                    for file in temp_files:
                        if os.path.exists(file) and os.path.getsize(file) > 0:
                            audio = AudioSegment.from_mp3(file)
                            combined += audio
                    
                    # Export the combined audio
                    combined.export(output_file, format="mp3")
                    
                    # Clean up temp files
                    for file in temp_files:
                        if os.path.exists(file):
                            os.remove(file)
                    
                    print(f"Generated combined ground statement audio: {output_file}")
                    return
            except Exception as e:
                print(f"Error in chunked processing: {e}, falling back to standard method")
    
    # Standard processing for non-ground statements or fallback for ground statements
    for attempt in range(1, max_retries + 1):
        try:
            # Create a new communicate object with proper rate setting
            rate = "-10%" if is_ground_statement else "-10%"  # Changed from "0%" to "-10%" for ground statements
            communicate = edge_tts.Communicate(text=escaped_text, voice=voice, rate=rate)
            await communicate.save(output_file)
            print(f"Generated audio: {output_file}")
            # Reduced sleep time after successful generation
            await asyncio.sleep(0.5)
            break
        except edge_tts.exceptions.NoAudioReceived as e:
            print(f"Attempt {attempt} failed: No audio received. Retrying...")
            if is_ground_statement:
                # Try alternative voice for ground statements
                if attempt == max_retries - 1:
                    print("Trying backup voice for ground statement...")
                    try:
                        communicate = edge_tts.Communicate(text=escaped_text, voice=VOICE_BACKUP, rate="-10%")  # Changed from "-5%" to "-10%"
                        await communicate.save(output_file)
                        print(f"Generated audio with backup voice: {output_file}")
                        break
                    except Exception as e2:
                        print(f"Backup voice approach failed: {str(e2)}")
            if attempt == max_retries:
                raise
            # Reduced sleep time between retry attempts
            await asyncio.sleep(0.5)

def verify_audio_file(file_path: str) -> bool:
    """Verify that an audio file exists and is valid."""
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return False
    
    if os.path.getsize(file_path) == 0:
        print(f"File is empty: {file_path}")
        return False
    
    try:
        # Try to load a small portion of the file to verify it's a valid audio file
        audio = AudioSegment.from_mp3(file_path)[:100]
        return True
    except Exception as e:
        print(f"Error verifying audio file {file_path}: {e}")
        return False

def combine_audio_files(files: List[str], output_file: str = "outputs/debate.mp3") -> None:
    """Combine multiple MP3 files into a single file."""
    if not files:
        print("No audio files to combine.")
        return
    
    # Verify each file before attempting to combine
    valid_files = [f for f in files if verify_audio_file(f)]
    
    if not valid_files:
        print("No valid audio files to combine.")
        return
    
    if len(valid_files) < len(files):
        print(f"Warning: {len(files) - len(valid_files)} files were skipped due to validation issues.")
    
    try:
        combined = AudioSegment.empty()
        for file in valid_files:
            try:
                print(f"Adding file to combined audio: {file}")
                audio = AudioSegment.from_mp3(file)
                combined += audio
                # Add a small pause between segments
                combined += AudioSegment.silent(duration=500)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
        
        # Ensure we have some audio to save
        if len(combined) > 0:
            combined.export(output_file, format="mp3")
            print(f"Combined audio saved as {output_file}")
        else:
            print("No audio was combined. Output file not created.")
    except Exception as e:
        print(f"Error combining audio files with pydub: {e}")
        
        # Fallback: try using ffmpeg directly
        try:
            print("Attempting to combine files using ffmpeg directly...")
            
            # Create a text file listing all the input files
            list_file = "audio_files_list.txt"
            with open(list_file, "w") as f:
                for file in valid_files:
                    f.write(f"file '{file}'\n")
            
            # Run ffmpeg command
            ffmpeg_cmd = f'ffmpeg -f concat -safe 0 -i {list_file} -c copy {output_file}'
            subprocess.run(ffmpeg_cmd, shell=True, check=True)
            print(f"Combined audio saved as {output_file} using ffmpeg")
            
            # Clean up the temporary file
            os.remove(list_file)
        except Exception as ffmpeg_error:
            print(f"Ffmpeg fallback also failed: {ffmpeg_error}")

async def process_debate() -> None:
    """Process the debate file and generate audio for each line."""
    # Clear existing audio files
    clear_output_folder()

    try:
        # Read the debate file
        with open('outputs/debate.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tasks: List[Tuple[str, str, str, bool]] = []
        output_files = []

        # Process each line
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Extract text with appropriate handling
            if line.startswith("Ground Statement:"):
                voice = VOICE_NARRATOR
                text = line
                is_ground = True
            elif line.startswith("AI Debater 1:"):
                voice = VOICE_JANE
                text = line.replace("AI Debater 1:", "").strip()
                is_ground = False
            elif line.startswith("AI Debater 2:"):
                voice = VOICE_VALENTINO
                text = line.replace("AI Debater 2:", "").strip()
                is_ground = False
            elif line.startswith("Result:"):
                voice = VOICE_NARRATOR
                text = line
                is_ground = False
            else:
                continue

            output_file = f"outputs/audio_output/{timestamp}_part_{i:02d}.mp3"
            output_files.append(output_file)
            tasks.append((text, voice, output_file, is_ground))

        # Process speech synthesis sequentially
        print("\nGenerating audio files...")
        for text, voice, output_file, is_ground in tasks:
            print(f"\nProcessing: {text[:50]}...")
            try:
                await text_to_speech(text, voice, output_file, is_ground_statement=is_ground)
                # Reduced sleep time after text_to_speech to ensure file is written
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"Warning: Failed to generate audio for text: {text[:50]}..., skipping. Error: {str(e)}")
        
        # Combine all audio files into a single file
        combine_audio_files(output_files)
        
        print(f"\nAll audio files have been created in the audio_output directory (timestamp: {timestamp})")
        print(f"Combined audio file has been saved as debate.mp3")
    
    except Exception as e:
        print(f"Error in process_debate: {str(e)}")
        raise
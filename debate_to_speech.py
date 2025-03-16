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
from openai import OpenAI
import json
from speech_to_text_timing import generate_accurate_timing

# Using standard Edge TTS voices
VOICE_NARRATOR = "en-US-ChristopherNeural"
VOICE_JANE = "en-US-JennyNeural"
VOICE_VALENTINO = "en-US-GuyNeural"
VOICE_BACKUP = "en-US-EricNeural"

def clear_output_folders():
    """Clear the audio_output and temp_frames folders if they exist."""
    # Clear audio output
    audio_output = 'outputs/audio_output'
    if os.path.exists(audio_output):
        shutil.rmtree(audio_output)
    os.makedirs(audio_output, exist_ok=True)
    
    # Clear temp frames
    temp_frames = 'outputs/temp_frames'
    if os.path.exists(temp_frames):
        shutil.rmtree(temp_frames)
    os.makedirs(temp_frames, exist_ok=True)

def escape_text(text: str) -> str:
    """Escape special characters in text for SSML."""
    return xml.sax.saxutils.escape(text)

def split_long_text(text: str, max_length: int = 1000) -> List[str]:
    """Split long text into smaller chunks at sentence boundaries."""
    if len(text) <= max_length:
        return [text]
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def text_to_speech(text: str, voice: str, output_file: str, is_ground_statement: bool = False) -> None:
    """Convert text to speech using Edge TTS with speed control."""
    max_retries = 3
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    escaped_text = escape_text(text)
    original_text = text  # Store original text before escaping
    
    if is_ground_statement and len(escaped_text) > 1000:
        chunks = split_long_text(escaped_text)
        temp_files = []
        
        try:
            for i, chunk in enumerate(chunks):
                temp_file = f"{output_file}.part{i}.mp3"
                temp_files.append(temp_file)
                
                try:
                    communicate = edge_tts.Communicate(text=chunk, voice=voice, rate="-10%")
                    await communicate.save(temp_file)
                except Exception:
                    communicate = edge_tts.Communicate(text=chunk, voice=VOICE_BACKUP, rate="-10%")
                    await communicate.save(temp_file)
            
            if temp_files:
                combined = AudioSegment.empty()
                for file in temp_files:
                    if os.path.exists(file) and os.path.getsize(file) > 0:
                        audio = AudioSegment.from_mp3(file)
                        combined += audio
                
                combined.export(output_file, format="mp3")
                
                for file in temp_files:
                    if os.path.exists(file):
                        os.remove(file)
                        
                # Generate timing info with original text
                try:
                    timing_info = generate_accurate_timing(output_file, original_text)
                except Exception:
                    pass
                        
                return
        except Exception:
            pass
    
    for attempt in range(1, max_retries + 1):
        try:
            rate = "-10%" if is_ground_statement else "-10%"
            communicate = edge_tts.Communicate(text=escaped_text, voice=voice, rate=rate)
            await communicate.save(output_file)
            
            # Generate timing info with original text
            try:
                timing_info = generate_accurate_timing(output_file, original_text)
            except Exception:
                pass
            
            break
        except edge_tts.exceptions.NoAudioReceived:
            if is_ground_statement and attempt == max_retries - 1:
                try:
                    communicate = edge_tts.Communicate(text=escaped_text, voice=VOICE_BACKUP, rate="-10%")
                    await communicate.save(output_file)
                    break
                except Exception:
                    pass
            if attempt == max_retries:
                raise

def verify_audio_file(file_path: str) -> bool:
    """Verify that an audio file exists and is valid."""
    if not os.path.exists(file_path):
        return False
    
    if os.path.getsize(file_path) == 0:
        return False
    
    try:
        audio = AudioSegment.from_mp3(file_path)[:100]
        return True
    except Exception:
        return False

def combine_audio_files(files: List[str], output_file: str = "outputs/debate.mp3") -> None:
    """Combine multiple MP3 files into a single file."""
    if not files:
        return
    
    valid_files = [f for f in files if verify_audio_file(f)]
    
    if not valid_files:
        return
    
    try:
        combined = AudioSegment.empty()
        for file in valid_files:
            try:
                audio = AudioSegment.from_mp3(file)
                combined += audio
                combined += AudioSegment.silent(duration=500)
            except Exception:
                continue
        
        if len(combined) > 0:
            combined.export(output_file, format="mp3")
    except Exception:
        try:
            list_file = "audio_files_list.txt"
            with open(list_file, "w") as f:
                for file in valid_files:
                    f.write(f"file '{file}'\n")
            
            ffmpeg_cmd = f'ffmpeg -f concat -safe 0 -i {list_file} -c copy {output_file}'
            subprocess.run(ffmpeg_cmd, shell=True, check=True)
            
            os.remove(list_file)
        except Exception:
            pass

def get_voice_for_speaker(speaker):
    """Get the appropriate voice for a given speaker."""
    if speaker == "Ground":
        return VOICE_NARRATOR  # Changed from Guy's voice to the narrator's voice
    elif speaker == "Jane":
        return VOICE_JANE
    elif speaker == "Valentino":
        return VOICE_VALENTINO
    elif speaker == "Narrator":
        return VOICE_NARRATOR
    elif speaker == "Result":
        return VOICE_NARRATOR  # Explicitly set Result to use narrator voice too
    else:
        return VOICE_BACKUP

async def process_debate():
    """Process the debate.txt file and convert text to speech."""
    try:
        # Clear output folders at the start
        clear_output_folders()
        
        segments = []
        with open('outputs/debate.txt', 'r', encoding='utf-8') as f:
            current_speaker = None
            current_text = []
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("Narrator:") or line.startswith("Ground Statement:") or \
                   line.startswith("AI Debater 1:") or line.startswith("AI Debater 2:") or \
                   line.startswith("Result:"):
                    if line.startswith("Narrator:"):
                        speaker = "Narrator"
                        text = line.replace("Narrator:", "").strip()
                    elif line.startswith("Ground Statement:"):
                        speaker = "Narrator"  # Changed from "Ground" to "Narrator"
                        text = line  # Keep the full line
                    elif line.startswith("AI Debater 1:"):
                        speaker = "Jane"
                        text = line.replace("AI Debater 1:", "").strip()
                    elif line.startswith("AI Debater 2:"):
                        speaker = "Valentino"
                        text = line.replace("AI Debater 2:", "").strip()
                    else:  # Result
                        speaker = "Narrator"  # Changed from "Result" to "Narrator"
                        text = line  # Keep the full line
                        
                    segments.append({"speaker": speaker, "text": text})
        
        os.makedirs('outputs/audio_output', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_files = []
        
        for i, segment in enumerate(segments):
            speaker = segment["speaker"]
            text = segment["text"]
            voice = get_voice_for_speaker(speaker)
            is_ground = "Ground Statement:" in text  # Check if it's ground statement by content
            
            output_file = f"outputs/audio_output/{timestamp}_part_{i:02d}.mp3"
            await text_to_speech(text, voice, output_file, is_ground_statement=is_ground)
            audio_files.append(output_file)
        
        combine_audio_files(audio_files)
        
    except Exception as e:
        raise e
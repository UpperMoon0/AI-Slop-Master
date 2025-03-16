import os
import time
import uuid
import json
import re
import logging
import datetime
from pathlib import Path
import edge_tts
import asyncio
from pydub import AudioSegment
from debate_to_video import split_text_into_smaller_parts, parse_debate_file
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Voice settings
VOICES = {
    'Narrator': 'en-US-EricNeural',
    'Jane': 'en-US-JennyNeural',
    'Valentino': 'en-US-GuyNeural'
}

# Output directory for audio files
OUTPUT_DIR = 'outputs/audio_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def text_to_speech(text: str, voice: str, output_file: str) -> bool:
    """Convert text to speech using Edge TTS."""
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        return True
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return False

async def process_debate_segments(segments: List[Dict[str, str]]) -> bool:
    """Process debate segments and generate audio."""
    try:
        os.makedirs('outputs/audio_output', exist_ok=True)
        
        for i, segment in enumerate(segments):
            speaker = segment["speaker"]
            text = segment["text"]
            
            # Select voice based on speaker
            voice = "en-US-JennyNeural" if speaker == "Jane" else "en-GB-RyanNeural"
            if speaker == "Narrator":
                voice = "en-US-ChristopherNeural"
            
            # Create audio file
            output_file = f'outputs/audio_output/part_{i:02d}.mp3'
            success = await text_to_speech(text, voice, output_file)
            
            if not success:
                print(f"Failed to create audio for segment {i+1}")
                return False
        
        return True
    except Exception as e:
        print(f"Error in process_debate_segments: {e}")
        return False

async def generate_debate_speech(segments: List[Dict[str, str]]) -> bool:
    """Generate speech for all debate segments."""
    try:
        success = await process_debate_segments(segments)
        return success
    except Exception as e:
        print(f"Error in generate_debate_speech: {e}")
        return False

async def process_debate() -> bool:
    """Process debate file and generate speech."""
    try:
        # Read debate file
        segments = []
        with open('outputs/debate.txt', 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                if ':' in line:
                    speaker, text = line.split(':', 1)
                    segments.append({
                        "speaker": speaker.strip(),
                        "text": text.strip()
                    })
        
        # Generate speech for segments
        success = await generate_debate_speech(segments)
        return success
    except Exception as e:
        print(f"Error in process_debate: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(process_debate())
    if success:
        print("Debate speech generation completed successfully!")
    else:
        print("Failed to generate debate speech.")
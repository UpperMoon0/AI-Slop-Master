import asyncio
import os
from typing import List, Tuple
import edge_tts
import wave
from datetime import datetime

VOICE_NARRATOR = "en-US-DavidNeural"
VOICE_AI1 = "en-US-JennyNeural"
VOICE_AI2 = "en-US-GuyNeural"

async def text_to_speech(text: str, voice: str, output_file: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

async def process_debate() -> None:
    # Read the debate file
    with open('debate.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Create output directory if it doesn't exist
    os.makedirs('audio_output', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each line and create audio files
    tasks: List[Tuple[str, str, str]] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if line.startswith("Ground Statement:"):
            voice = VOICE_NARRATOR
            text = line.replace("Ground Statement:", "").strip()
        elif line.startswith("AI Debater 1:"):
            voice = VOICE_AI1
            text = line.replace("AI Debater 1:", "").strip()
        elif line.startswith("AI Debater 2:"):
            voice = VOICE_AI2
            text = line.replace("AI Debater 2:", "").strip()
        else:
            continue

        output_file = f"audio_output/{timestamp}_part_{i:02d}.mp3"
        tasks.append((text, voice, output_file))

    # Convert text to speech in parallel
    await asyncio.gather(
        *(text_to_speech(text, voice, output_file) 
          for text, voice, output_file in tasks)
    )
    
    print(f"Audio files have been created in the audio_output directory with timestamp {timestamp}")

if __name__ == "__main__":
    asyncio.run(process_debate())
#!/usr/bin/env python3
"""
Example script demonstrating how to use Edge TTS voices with the updated TTS system.
This example generates speech samples for all available Edge TTS voices.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.audio_utils import text_to_speech, VOICES

# Demo text to generate for each voice
DEMO_TEXT = "Hello there! I'm a text-to-speech voice from Microsoft Edge TTS. I sound much more natural than traditional TTS voices."
DEMO_TEXT_LONG = """
Welcome to this demonstration of the Edge TTS voices. 
These voices offer high-quality, natural-sounding speech synthesis.
They can be used for a variety of applications including audiobook narration,
virtual assistants, and accessibility tools for the visually impaired.
The Edge TTS engine also supports adjustments to speech rate, volume, and pitch,
allowing for fine-tuning of the voice characteristics to suit your needs.
"""

async def main():
    # Create output directory
    output_dir = "outputs/edge_tts_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all Edge TTS voice configurations
    edge_voices = {name: config for name, config in VOICES.items() 
                  if config.get("model") == "edge"}
    
    print(f"Found {len(edge_voices)} Edge TTS voices")
    
    # Generate speech for each voice
    for voice_name, voice_config in edge_voices.items():
        print(f"\nGenerating speech for {voice_name}")
        
        # Standard sample
        output_file = os.path.join(output_dir, f"{voice_name}_standard.wav")
        await text_to_speech(DEMO_TEXT, voice_name, output_file)
        
        # Sample with faster rate
        voice_config_fast = voice_config.copy()
        voice_config_fast["rate"] = "+20%"
        voice_config_fast["volume"] = "+10%"
        
        # Create a temporary voice entry
        VOICES[f"{voice_name}_fast"] = voice_config_fast
        
        output_file = os.path.join(output_dir, f"{voice_name}_fast.wav")
        await text_to_speech(DEMO_TEXT, f"{voice_name}_fast", output_file)
        
        # Sample with slower rate and lower pitch
        voice_config_slow = voice_config.copy()
        voice_config_slow["rate"] = "-20%"
        voice_config_slow["pitch"] = "-10%"
        
        # Create a temporary voice entry
        VOICES[f"{voice_name}_slow"] = voice_config_slow
        
        output_file = os.path.join(output_dir, f"{voice_name}_slow.wav")
        await text_to_speech(DEMO_TEXT, f"{voice_name}_slow", output_file)
        
        # Longer text sample
        output_file = os.path.join(output_dir, f"{voice_name}_long.wav")
        await text_to_speech(DEMO_TEXT_LONG, voice_name, output_file)
    
    print("\nSpeech generation complete. Audio files saved to:", output_dir)
    print("Available files:")
    
    for file in sorted(os.listdir(output_dir)):
        if file.endswith(".wav"):
            file_path = os.path.join(output_dir, file)
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  - {file} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    asyncio.run(main()) 
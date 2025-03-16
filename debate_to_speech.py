import os
import logging
import asyncio
from utils.file_utils import parse_debate_file
from utils.audio_utils import generate_debate_speech

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output directory for audio files
OUTPUT_DIR = 'outputs/audio_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def process_debate() -> bool:
    """Process debate file and generate speech."""
    try:
        # Use the existing parse_debate_file utility function
        segments = parse_debate_file()
        
        # Use the utility function to generate speech
        success = await generate_debate_speech(segments, OUTPUT_DIR)
        return success
    except Exception as e:
        logger.error(f"Error in process_debate: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(process_debate())
    if success:
        logger.info("Debate speech generation completed successfully!")
    else:
        logger.error("Failed to generate debate speech.")
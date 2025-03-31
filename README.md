# AI-Slop-Master

A system for generating AI debates with speech synthesis and video generation.

## Features

- Generate AI debates between multiple characters
- Convert text debates to speech using multiple TTS engines:
  - Sesame CSM-1B (high-quality but resource-intensive)
  - Microsoft Edge TTS (excellent quality with lower resource usage)
- Create videos with synchronized audio and subtitles
- Support for various voice configurations with adjustable speech parameters

## Setup

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/AI-Slop-Master.git
   cd AI-Slop-Master
   ```

2. Create and activate a virtual environment (Python 3.8+ recommended)
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/Mac:
   source .venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up TTS-Provider (optional if you want to use Sesame CSM-1B)
   ```bash
   cd TTS-Provider
   pip install -r requirements.txt
   ```

## Using Edge TTS

The system now supports Microsoft Edge TTS, which provides high-quality natural-sounding voices without requiring a GPU.

Benefits of Edge TTS:
- Much faster than neural TTS models (near real-time)
- No GPU required
- Multiple languages and voices available
- Adjustable speech parameters (rate, volume, pitch)
- Higher reliability for long texts

To use Edge TTS voices in your debates, simply choose one of the Edge voice configurations:
- "Jenny" - US Female
- "Guy" - US Male
- "Aria" - US Female (professional)
- "Ryan" - UK Male
- "Sonia" - UK Female

### Edge TTS Demo

You can run a demo to test all Edge TTS voices:

```bash
python examples/edge_tts_example.py
```

This will generate sample audio files for each Edge TTS voice with different speech parameters.

## Running the TTS Server

The TTS server supports both Sesame CSM-1B and Edge TTS models.

Start the server:
```bash
cd TTS-Provider
python -m run_server
```

The server will run on `ws://localhost:9000` by default.

## Generating Debates

1. Create a debate text file in the required format
2. Run the debate to speech conversion
   ```bash
   python debate_to_speech.py
   ```

3. Generate a video from the audio (optional)
   ```bash
   python debate_to_video.py
   ```

## Voice Configuration

You can customize voices in `utils/audio_utils.py` by modifying the `VOICES` dictionary. For Edge TTS, you can adjust:
- Rate (e.g., "+10%", "-20%")
- Volume (e.g., "+20%", "-10%")
- Pitch (e.g., "+5%", "-15%")

Example voice configuration:
```python
"MyCustomVoice": {
    "speaker": 1, 
    "model": "edge", 
    "rate": "+5%", 
    "volume": "+10%", 
    "pitch": "-5%"
}
```

## License

[MIT License](LICENSE) 
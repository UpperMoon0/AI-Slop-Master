import os
import json
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

def generate_accurate_timing(audio_file, text, output_file=None):
    """
    Generate accurate timing for subtitles using speech recognition.
    This function can be used as an alternative to the estimation-based approach.
    
    Args:
        audio_file: Path to the MP3 audio file
        text: The text spoken in the audio file (original text from debate.txt)
        output_file: Path to save timing JSON (defaults to audio_file with _timing.json)
    
    Returns:
        Dictionary containing timing segments
    """
    if not output_file:
        output_file = audio_file.replace('.mp3', '_timing.json')
    
    print(f"Analyzing audio timing for {os.path.basename(audio_file)}...")
    
    try:
        # Load audio using pydub
        audio = AudioSegment.from_mp3(audio_file)
        
        # Convert to WAV for speech recognition
        wav_file = audio_file.replace('.mp3', '_temp.wav')
        audio.export(wav_file, format="wav")
        
        # Initialize speech recognition
        r = sr.Recognizer()
        
        # Split audio on silences to get better segments
        chunks = split_on_silence(
            audio, 
            min_silence_len=500,  # minimum silence length in ms
            silence_thresh=-40,   # consider anything quieter than this silence
            keep_silence=300      # keep some silence at the beginning and end
        )
        
        timing_segments = []
        current_time = 0
        
        # Process each audio chunk
        for i, chunk in enumerate(chunks):
            # Save chunk to temporary file
            chunk_file = f"temp_chunk_{i}.wav"
            chunk.export(chunk_file, format="wav")
            
            # Get duration of chunk
            chunk_duration = len(chunk) / 1000.0
            
            # Perform speech recognition on chunk
            with sr.AudioFile(chunk_file) as source:
                audio_data = r.record(source)
                try:
                    # Try to recognize text in this chunk (for timing purposes only)
                    recognized_text = r.recognize_google(audio_data)
                    
                    # Create timing segment with both recognized and original text
                    timing_segments.append({
                        "text": text,  # Use original text from debate.txt
                        "recognized_text": recognized_text,  # Store recognized text for reference
                        "start_time": current_time,
                        "end_time": current_time + chunk_duration
                    })
                except sr.UnknownValueError:
                    # No speech detected in this chunk (might be silence)
                    pass
                except Exception as e:
                    print(f"Error processing chunk {i}: {e}")
            
            # Update current time position
            current_time += chunk_duration
            
            # Clean up temporary file
            os.remove(chunk_file)
        
        # Clean up WAV file
        os.remove(wav_file)
        
        # If we have no segments, create a single segment with the full duration
        if not timing_segments and text:
            timing_segments.append({
                "text": text,
                "recognized_text": "",
                "start_time": 0,
                "end_time": len(audio) / 1000.0
            })
        
        # Save timing data
        timing_data = {"segments": timing_segments}
        with open(output_file, 'w') as f:
            json.dump(timing_data, f, indent=2)
        
        return timing_data
        
    except Exception as e:
        print(f"Error generating timing: {e}")
        # Create a fallback timing with entire duration
        if os.path.exists(audio_file):
            try:
                audio = AudioSegment.from_mp3(audio_file)
                duration = len(audio) / 1000.0
                timing_data = {
                    "segments": [{
                        "text": text,
                        "recognized_text": "",
                        "start_time": 0,
                        "end_time": duration
                    }]
                }
                with open(output_file, 'w') as f:
                    json.dump(timing_data, f, indent=2)
                return timing_data
            except:
                pass
        return {"segments": []}
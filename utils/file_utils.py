import os

def parse_debate_file():
    """Parse debate.txt file to get dialogue segments and speakers."""
    dialogue_segments = []
    
    try:
        with open('outputs/debate.txt', 'r', encoding='utf-8') as f:
            current_speaker = None
            current_text = ""
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Skip Display Summary line - it should only be displayed, not spoken
                if line.startswith("Display Summary:"):
                    continue
                
                new_speaker = None
                text = line
                
                # Determine speaker from line prefix
                if line.startswith("Narrator:"):
                    new_speaker = "Narrator"
                    text = line.replace("Narrator:", "").strip()
                elif line.startswith("Ground Statement:") or line.startswith("Result:"):
                    new_speaker = "Narrator"
                    text = line.strip()
                elif line.startswith("AI Debater 1:"):
                    new_speaker = "Jane"  # AI Debater 1 is always Jane
                    text = line.replace("AI Debater 1:", "").strip()
                elif line.startswith("AI Debater 2:"):
                    new_speaker = "Valentino"  # AI Debater 2 is always Valentino
                    text = line.replace("AI Debater 2:", "").strip()
                else:
                    continue
                
                # If we found a speaker and text
                if new_speaker and text:
                    # Add the completed previous segment if speaker changes
                    if current_speaker and (current_speaker != new_speaker or current_speaker == "Narrator"):
                        if current_text:
                            dialogue_segments.append({"speaker": current_speaker, "text": current_text})
                        current_text = ""
                    
                    # Update current speaker and add text
                    current_speaker = new_speaker
                    current_text = text if not current_text else current_text + " " + text
            
            # Add the final segment
            if current_speaker and current_text:
                dialogue_segments.append({"speaker": current_speaker, "text": current_text})
    
    except (FileNotFoundError, IOError) as e:
        print(f"Error reading debate file: {str(e)}")
    
    # Validate segments to ensure consistent speaker attribution
    validated_segments = []
    for i, segment in enumerate(dialogue_segments):
        if i > 0 and segment["speaker"] == dialogue_segments[i-1]["speaker"]:
            # If same speaker in consecutive segments, merge them
            if validated_segments:  # Make sure we have at least one segment
                validated_segments[-1]["text"] += " " + segment["text"]
        else:
            validated_segments.append(segment)
    
    print(f"Parsed {len(validated_segments)} dialogue segments")
    return validated_segments

def cleanup_temp_files(temp_frames_dir, project_temp_dir):
    """Clean up all temporary files after video creation."""
    print("Cleaning up temporary files...")
    
    try:
        # Clean up frames directory
        if os.path.exists(temp_frames_dir):
            for filename in os.listdir(temp_frames_dir):
                file_path = os.path.join(temp_frames_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        # Clean MoviePy temp directory
        if os.path.exists(project_temp_dir):
            for filename in os.listdir(project_temp_dir):
                file_path = os.path.join(project_temp_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        os.unlink(file_path)
                    except Exception as e:
                        print(f"Error cleaning temp file {file_path}: {e}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

import os
import time
import shutil

# Add a global variable to store the summary
_ground_statement_summary = None

def parse_debate_file():
    """Parse debate.txt file to get dialogue segments and speakers."""
    global _ground_statement_summary
    dialogue_segments = []
    
    try:
        with open('outputs/debate.txt', 'r', encoding='utf-8') as f:
            current_speaker = None
            current_text = ""
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Extract the summary but don't include it in spoken dialogue
                if line.startswith("Summary:"):
                    _ground_statement_summary = line.replace("Summary:", "").strip()
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

def get_ground_statement_summary():
    """Get the extracted ground statement summary."""
    global _ground_statement_summary
    
    # If we don't have a summary yet, try to extract it directly
    if _ground_statement_summary is None:
        try:
            with open('outputs/debate.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith("Summary:"):
                        _ground_statement_summary = line.replace("Summary:", "").strip()
                        break
        except Exception as e:
            print(f"Error extracting ground statement summary: {e}")
    
    return _ground_statement_summary

def cleanup_temp_files(temp_frames_dir, project_temp_dir):
    """Clean up all temporary files after video creation."""
    print("Cleaning up temporary files...")
    
    # Allow a short delay to ensure files are fully released
    time.sleep(1)
    
    try:
        # Clean up frames directory
        if os.path.exists(temp_frames_dir):
            for filename in os.listdir(temp_frames_dir):
                file_path = os.path.join(temp_frames_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        # Try with multiple retries for locked files
                        delete_with_retry(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            
            # Try to remove the directory itself if empty
            try:
                # Just in case there are any leftover files, attempt to remove the whole directory
                if not os.listdir(temp_frames_dir):
                    os.rmdir(temp_frames_dir)
            except Exception as e:
                print(f"Could not remove temp directory {temp_frames_dir}: {e}")
        
        # Clean MoviePy temp directory
        if os.path.exists(project_temp_dir):
            for filename in os.listdir(project_temp_dir):
                file_path = os.path.join(project_temp_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        delete_with_retry(file_path)
                    except Exception as e:
                        print(f"Error cleaning temp file {file_path}: {e}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def delete_with_retry(file_path, max_attempts=3, delay=1):
    """Try to delete a file with multiple retries if it's locked."""
    for attempt in range(max_attempts):
        try:
            os.unlink(file_path)
            return True
        except PermissionError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
                continue
            else:
                print(f"File {file_path} is locked, will be removed on next run.")
                return False
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
            return False

def reformat_debate_file(file_path='outputs/debate.txt'):
    """
    Reformat the debate.txt file to merge multi-line arguments into single lines.
    Only keeps line breaks before official speaker/section identifiers.
    
    Args:
        file_path: Path to the debate.txt file
        
    Returns:
        bool: True if reformatting was successful, False otherwise
    """
    valid_prefixes = [
        "Narrator:", 
        "Ground Statement:", 
        "Summary:", 
        "AI Debater 1:", 
        "AI Debater 2:", 
        "Result:"
    ]
    
    try:
        # Read the entire file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into lines and process
        lines = content.split('\n')
        reformatted_lines = []
        current_line = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
                
            # Check if this line starts with a valid prefix
            is_valid_start = any(line.startswith(prefix) for prefix in valid_prefixes)
            
            if is_valid_start:
                # If we have accumulated content from previous speaker, save it
                if current_line:
                    reformatted_lines.append(current_line)
                # Start a new line with this prefix
                current_line = line
            else:
                # This is a continuation of the previous speaker's text
                # Add a space and append to the current line
                if current_line:
                    current_line += " " + line
                else:
                    # Shouldn't normally happen, but handle just in case
                    current_line = line
        
        # Add the last accumulated line if it exists
        if current_line:
            reformatted_lines.append(current_line)
        
        # Join all reformatted lines with newlines and write back to the file
        reformatted_content = '\n'.join(reformatted_lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(reformatted_content)
        
        print(f"Successfully reformatted {file_path}")
        return True
    except Exception as e:
        print(f"Error reformatting debate file: {str(e)}")
        return False
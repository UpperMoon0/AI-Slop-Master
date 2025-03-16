from PIL import ImageFont

def get_font_metrics(font, text):
    """Get font metrics in a way that works with both old and new Pillow versions."""
    try:
        # Try new Pillow methods first
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]  # width, height
    except AttributeError:
        try:
            # Try getlength for width (intermediate Pillow versions)
            width = font.getlength(text)
            # For height, we'll use a reference character
            _, height = font.getsize("Tg")
            return width, height
        except AttributeError:
            # Fall back to old getsize method
            return font.getsize(text)

def wrap_text(text, font, max_width):
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        # Try adding the word to the current line
        test_line = current_line + [word]
        test_text = " ".join(test_line)
        text_width, _ = get_font_metrics(font, test_text)
            
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:  # Add the current line if it's not empty
                lines.append(" ".join(current_line))
            current_line = [word]
    
    if current_line:  # Add the last line
        lines.append(" ".join(current_line))
    
    return lines

def split_text_into_smaller_parts(text):
    """
    Split text into smaller parts (sentences or parts of sentences) for better subtitle timing.
    
    Returns a list of text chunks optimized for subtitle display.
    """
    # First, split by sentence endings (., !, ?)
    sentence_delimiters = ['. ', '! ', '? ']
    sentences = []
    current_text = text
    
    # Extract sentences
    for delimiter in sentence_delimiters:
        parts = current_text.split(delimiter)
        for i in range(len(parts) - 1):
            sentences.append(parts[i] + delimiter.rstrip())
        current_text = parts[-1]
    
    # Add the last part if it's not empty
    if current_text.strip():
        sentences.append(current_text.strip())
    
    # Now further split long sentences
    max_words_per_segment = 8  # Smaller chunks for better subtitle timing
    final_segments = []
    
    for sentence in sentences:
        words = sentence.split()
        
        if len(words) <= max_words_per_segment:
            final_segments.append(sentence)
        else:
            # Split into smaller chunks based on commas or just word count
            # Try to split by commas first
            comma_parts = sentence.split(', ')
            temp_parts = []
            
            for part in comma_parts:
                part_words = part.split()
                if len(part_words) <= max_words_per_segment:
                    temp_parts.append(part)
                else:
                    # Further split by word count if still too long
                    for i in range(0, len(part_words), max_words_per_segment):
                        chunk = part_words[i:i + max_words_per_segment]
                        temp_parts.append(' '.join(chunk))
            
            # Add commas back except for the last part
            for i in range(len(temp_parts) - 1):
                final_segments.append(temp_parts[i] + ',')
            final_segments.append(temp_parts[-1])
    
    # Ensure no empty segments
    final_segments = [segment for segment in final_segments if segment.strip()]
    
    return final_segments

def split_text_into_chunks(text, chunk_size=8):
    """Split text into chunks of roughly equal size."""
    if not text:  # Handle empty text
        return [""]
        
    words = text.split()
    if not words:  # Handle text with no words
        return [""]
        
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    
    if current_chunk:  # Add any remaining words
        chunks.append(" ".join(current_chunk))
    
    return chunks if chunks else [text]  # Return original text if no chunks were created

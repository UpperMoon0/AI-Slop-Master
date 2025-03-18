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

def split_text_into_smaller_parts(text: str, max_chars: int = 40) -> list:
    """
    Split text into smaller parts (sentences or parts of sentences) for better subtitle timing.
    
    Args:
        text: The text to split
        max_chars: Maximum number of characters per segment that's
                  appropriate for subtitle display.
    
    Returns a list of text chunks optimized for subtitle display.
    """
    if not text:
        return []
    
    # First, try to split by sentences
    sentence_enders = ['. ', '! ', '? ', '.\n', '!\n', '?\n', ':"', '."']
    sentences = []
    last_end = 0
    
    # Find all sentence endings
    for i in range(len(text) - 1):
        for ender in sentence_enders:
            if i + len(ender) <= len(text) and text[i:i+len(ender)].startswith(ender):
                sentences.append(text[last_end:i+1])
                last_end = i + 1
                while last_end < len(text) and text[last_end] in [' ', '\n']:
                    last_end += 1
                break
    
    # Add the last part if there's any remaining text
    if last_end < len(text):
        sentences.append(text[last_end:])
    
    # If no sentences were found or we only have one sentence, split by commas, colons, or semicolons
    if len(sentences) <= 1:
        for sentence in sentences[:]:  # Use a copy of the list
            for separator in [', ', '; ', ': ']:
                parts = sentence.split(separator)
                if len(parts) > 1:
                    sentences = []
                    for i, part in enumerate(parts):
                        if i < len(parts) - 1:
                            sentences.append(part + separator)
                        else:
                            sentences.append(part)
                    break
    
    # If we still have long sentences, break them up further
    final_segments = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            final_segments.append(sentence)
        else:
            # Break up by words while respecting max_chars
            words = sentence.split(' ')
            current_segment = ""
            
            for word in words:
                if len(current_segment) + len(word) + 1 <= max_chars:
                    if current_segment:
                        current_segment += " " + word
                    else:
                        current_segment = word
                else:
                    # Current segment is full, start a new one
                    if current_segment:
                        final_segments.append(current_segment)
                    current_segment = word
            
            # Add the last segment if there's any remaining text
            if current_segment:
                final_segments.append(current_segment)
    
    # Ensure no empty segments
    final_segments = [segment.strip() for segment in final_segments if segment.strip()]
    
    # Additional pass to merge very short segments with neighbors if possible
    if len(final_segments) > 1:
        i = 0
        while i < len(final_segments) - 1:
            curr_segment = final_segments[i]
            next_segment = final_segments[i + 1]
            
            # If combining them would still be under max_chars, merge them
            if len(curr_segment) + len(next_segment) + 1 <= max_chars:
                final_segments[i] = curr_segment + " " + next_segment
                final_segments.pop(i + 1)
            else:
                i += 1
    
    return final_segments

def split_text_into_chunks(text, chunk_size=15):  # Increased from 8 to 15
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

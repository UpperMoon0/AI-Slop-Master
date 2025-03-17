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
    
    Creates longer, more natural chunks for better speaker tracking while still being
    appropriate for subtitle display.
    
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
    
    # Combine short sentences and split long ones to create balanced chunks
    max_words_per_segment = 20  # Increased from 8 to 20 for better chunking
    min_words_per_segment = 8   # New parameter for minimum chunk size
    
    final_segments = []
    current_segment = ""
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        sentence_word_count = len(words)
        
        # Case 1: If adding this sentence doesn't exceed max words, add it to current segment
        if current_word_count + sentence_word_count <= max_words_per_segment:
            current_segment += (" " if current_segment else "") + sentence
            current_word_count += sentence_word_count
            
            # If we've reached a good size, create a segment
            if current_word_count >= min_words_per_segment:
                final_segments.append(current_segment)
                current_segment = ""
                current_word_count = 0
                
        # Case 2: If this sentence alone is longer than max words, split it intelligently
        elif sentence_word_count > max_words_per_segment:
            # First add any existing content as a segment if it's not empty
            if current_segment:
                final_segments.append(current_segment)
                current_segment = ""
                current_word_count = 0
            
            # Try to split by natural breaks like commas first
            comma_parts = sentence.split(', ')
            
            if len(comma_parts) > 1:
                # Process comma-separated parts
                part_segment = ""
                part_word_count = 0
                
                for part_idx, part in enumerate(comma_parts):
                    part_words = part.split()
                    part_word_count_temp = len(part_words)
                    
                    # If adding this part doesn't exceed max, add it
                    if part_word_count + part_word_count_temp <= max_words_per_segment:
                        separator = ", " if part_segment else ""
                        part_segment += separator + part
                        part_word_count += part_word_count_temp
                    else:
                        # Finalize current part segment if it meets minimum size
                        if part_segment and part_word_count >= min_words_per_segment:
                            final_segments.append(part_segment)
                        
                        # Start new segment with current part
                        part_segment = part
                        part_word_count = part_word_count_temp
                
                # Add any remaining part segment
                if part_segment:
                    final_segments.append(part_segment)
            else:
                # If no natural breaks, split by word count but try to keep phrases together
                for i in range(0, sentence_word_count, max_words_per_segment):
                    chunk_words = words[i:min(i + max_words_per_segment, sentence_word_count)]
                    chunk_text = ' '.join(chunk_words)
                    final_segments.append(chunk_text)
        
        # Case 3: If adding this sentence would exceed max words but we have some content
        else:
            # Add the current segment if it meets minimum size
            if current_segment and current_word_count >= min_words_per_segment:
                final_segments.append(current_segment)
            
            # Start a new segment with this sentence
            current_segment = sentence
            current_word_count = sentence_word_count
    
    # Add any remaining content if it's not empty
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
            
            curr_words = len(curr_segment.split())
            next_words = len(next_segment.split())
            
            # If current segment is very short, try to merge with next
            if curr_words < min_words_per_segment and curr_words + next_words <= max_words_per_segment:
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

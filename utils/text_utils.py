def get_font_metrics(font, text):
    """Get the width and height of text using the most appropriate method
    for the version of PIL being used.
    
    Args:
        font: PIL ImageFont object
        text: Text string to measure
        
    Returns:
        Tuple of (width, height) in pixels
    """
    try:
        # New method in newer versions of PIL
        left, top, right, bottom = font.getbbox(text)
        return (right - left, bottom - top)
    except AttributeError:
        try:
            # getlength is available in some versions
            return (font.getlength(text), font.getsize(text)[1])
        except AttributeError:
            # Fall back to older getsize method
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

def split_text_into_chunks(text, max_chars=80):
    """Split text into chunks for timing purposes.
    
    Args:
        text: Text to split
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    # First, check if the text is short enough to use as-is
    if len(text) <= max_chars:
        return [text]
        
    # Try to split on sentence boundaries
    sentences = text.replace('. ', '.\n').replace('! ', '!\n').replace('? ', '?\n').split('\n')
    
    # Initialize result chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the limit, start a new chunk
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If the sentence itself is too long, split it by words
            if len(sentence) > max_chars:
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 > max_chars:
                        chunks.append(current_chunk)
                        current_chunk = word
                    else:
                        if current_chunk:
                            current_chunk += " " + word
                        else:
                            current_chunk = word
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

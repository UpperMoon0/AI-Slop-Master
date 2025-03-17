from PIL import ImageDraw
from utils.text_utils import wrap_text, get_font_metrics
from config import TEXT_COLOR, VIDEO_WIDTH, VIDEO_HEIGHT, TEXT_FONT

class Text:
    """A wrapper class for text displayed in videos."""
    
    def __init__(self, position="top", max_width=None, background=True, background_color=(220, 220, 220, 180)):
        """
        Initialize a Text container.
        
        Args:
            position (str): "top" or "bottom" for vertical positioning
            max_width (int, optional): Maximum width for text wrapping
            background (bool): Whether to draw background behind text
            background_color (tuple): RGB + alpha color for background
        """
        self.position = position.lower()
        self.max_width = max_width if max_width else (VIDEO_WIDTH - 300 if position == "top" else VIDEO_WIDTH - 100)
        self.background = background
        self.background_color = background_color
        self.text = None
        self.last_wrapped_lines = []
    
    def update_text(self, new_text):
        """Update the text content."""
        if self.text != new_text:
            self.text = new_text
            # Pre-compute wrapped lines
            if new_text:
                self.last_wrapped_lines = wrap_text(new_text, TEXT_FONT, self.max_width)
            else:
                self.last_wrapped_lines = []
        return self
    
    def clear(self):
        """Clear the text content."""
        self.text = None
        self.last_wrapped_lines = []
        return self
    
    def draw(self, draw_object):
        """
        Draw the text on the provided PIL ImageDraw object.
        
        Args:
            draw_object: PIL ImageDraw object
        """
        if not self.text or not self.last_wrapped_lines:
            return
        
        # Calculate total height of text block
        _, line_height = get_font_metrics(TEXT_FONT, "Tg")
        line_spacing = 10
        total_height = len(self.last_wrapped_lines) * (line_height + line_spacing)
        
        # Determine vertical position
        if self.position == "top":
            text_y = 20  # Top padding
        else:  # bottom
            subtitle_vertical_offset = 80  # Space from bottom edge
            text_y = VIDEO_HEIGHT - subtitle_vertical_offset - total_height
        
        # Draw each line centered horizontally
        for line in self.last_wrapped_lines:
            text_width, _ = get_font_metrics(TEXT_FONT, line)
            text_x = (VIDEO_WIDTH - text_width) // 2
            
            # Add background for text if enabled
            if self.background:
                text_bg_padding = 10
                
                # Draw rounded rectangle background
                draw_object.rectangle(
                    [(text_x - text_bg_padding, text_y - text_bg_padding/2),
                     (text_x + text_width + text_bg_padding, text_y + line_height + text_bg_padding/2)],
                    fill=self.background_color
                )
            
            # Draw the text
            draw_object.text((text_x, text_y), line, fill=TEXT_COLOR, font=TEXT_FONT)
            text_y += line_height + line_spacing

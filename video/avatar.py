import cv2
import numpy as np
from PIL import ImageDraw
from utils.text_utils import get_font_metrics
from config import NAME_FONT, TEXT_COLOR, HIGHLIGHT_COLOR

class Avatar:
    """A wrapper class for debater avatars displayed in videos."""
    
    def __init__(self, image, name, position, size=250):
        """
        Initialize an Avatar object.
        
        Args:
            image: OpenCV BGR image array for the avatar
            name: Debater name (string)
            position: (x, y) tuple for the top-left position
            size: Size of the avatar in pixels
        """
        self.image = image
        self.name = name
        self.position = position
        self.size = size
        self.highlighted = False
    
    def set_highlight(self, highlighted):
        """Set the highlighting state of the avatar."""
        self.highlighted = highlighted
        return self
    
    def draw_on_frame(self, frame):
        """
        Draw the avatar on the OpenCV frame.
        
        Args:
            frame: OpenCV BGR image array
        """
        # Calculate avatar position
        x, y = self.position
        
        # Draw the avatar on the frame
        if self.image is not None:
            # Make sure the avatar image is the right size
            avatar = cv2.resize(self.image, (self.size, self.size)) if self.image.shape[0] != self.size else self.image
            
            # Convert RGB to BGR if needed (fixing color inversion issue)
            if len(avatar.shape) == 3 and avatar.shape[2] == 3:
                avatar_bgr = cv2.cvtColor(avatar, cv2.COLOR_RGB2BGR)
            else:
                avatar_bgr = avatar
                
            # Copy the avatar to the frame
            frame[y:y+self.size, x:x+self.size] = avatar_bgr
        
        # Add highlight if needed
        if self.highlighted:
            highlight_thickness = 10
            # Fix the highlight color - HIGHLIGHT_COLOR is likely defined as RGB but OpenCV expects BGR
            highlight_color_bgr = (HIGHLIGHT_COLOR[2], HIGHLIGHT_COLOR[1], HIGHLIGHT_COLOR[0])
            
            cv2.rectangle(
                frame,
                (x-highlight_thickness, y-highlight_thickness),
                (x+self.size+highlight_thickness, y+self.size+highlight_thickness),
                highlight_color_bgr,  # Use BGR order for OpenCV
                highlight_thickness
            )
    
    def draw_name(self, draw_object):
        """
        Draw the name below the avatar using PIL ImageDraw.
        
        Args:
            draw_object: PIL ImageDraw object
        """
        x, y = self.position
        name_y = y + self.size + 10
        
        name_width, _ = get_font_metrics(NAME_FONT, self.name)
        name_x = x + (self.size - name_width) // 2
        
        draw_object.text((name_x, name_y), self.name, fill=TEXT_COLOR, font=NAME_FONT)

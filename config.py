import os
import cv2
import numpy as np
import moviepy.config as mpconfig
from PIL import ImageFont

# Configure MoviePy to use a custom temp folder to avoid filling up C drive
PROJECT_TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'moviepy_temp')
os.makedirs(PROJECT_TEMP_DIR, exist_ok=True)
mpconfig.change_settings({"TEMP_DIR": PROJECT_TEMP_DIR})

# Video generation constants
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30  # Updated from 24 to 30
BACKGROUND_COLOR = (240, 240, 240)  # RGB format
HIGHLIGHT_COLOR = (0, 223, 255)  # Use BGR format (255, 223, 0) in RGB for yellow highlighting
TEXT_COLOR = (0, 0, 0)  # RGB format
GROUND_STATEMENT_COLOR = (40, 40, 40)  # RGB format

# Cache fonts
try:
    NAME_FONT = ImageFont.truetype("arial.ttf", 36)
    TEXT_FONT = ImageFont.truetype("arial.ttf", 32)
except:
    NAME_FONT = TEXT_FONT = ImageFont.load_default()

# Cache avatar images
try:
    JANE_AVATAR = cv2.resize(cv2.imread('assets/jane_avatar.jpg'), (250, 250))
    VALENTINO_AVATAR = cv2.resize(cv2.imread('assets/valentino_avatar.jpg'), (250, 250))
except:
    JANE_AVATAR = VALENTINO_AVATAR = None

# Output paths
TEMP_FRAMES_DIR = os.path.join('outputs', 'temp_frames')
os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)

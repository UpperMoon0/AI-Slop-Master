import pytest
import numpy as np
import json
import os
from unittest.mock import Mock, patch, mock_open, PropertyMock
import cv2
from PIL import Image, ImageDraw, ImageFont
from debate_to_video import (
    create_frame, wrap_text, split_text_into_smaller_parts,
    parse_debate_file, get_segment_audio_file, get_segment_duration,
    get_segment_timing, create_debate_video
)

@pytest.fixture
def mock_fonts():
    """Mock font loading."""
    with patch('PIL.ImageFont.truetype') as mock_truetype:
        mock_font = Mock()
        mock_truetype.return_value = mock_font
        # Mock font metrics with proper return values
        mock_font.getbbox = Mock(return_value=(0, 0, 100, 30))
        mock_font.getsize = Mock(return_value=(100, 30))
        mock_font.getlength = Mock(return_value=100)  # Add getlength mock
        yield mock_font

@pytest.fixture
def mock_avatars():
    """Mock avatar loading."""
    mock_image = np.ones((250, 250, 3), dtype=np.uint8) * 255
    with patch('cv2.imread', return_value=mock_image):
        with patch('cv2.resize', return_value=mock_image):
            yield mock_image

def test_wrap_text(mock_fonts):
    """Test text wrapping functionality."""
    text = "This is a long text that needs to be wrapped properly"
    max_width = 200
    font = mock_fonts
    
    # Mock getlength to return increasing values for longer text
    def mock_get_length(text):
        return len(text) * 10
    font.getlength.side_effect = mock_get_length
    
    wrapped_lines = wrap_text(text, font, max_width)
    assert len(wrapped_lines) > 0
    assert isinstance(wrapped_lines, list)
    assert all(isinstance(line, str) for line in wrapped_lines)

def test_split_text_into_smaller_parts():
    """Test text splitting functionality."""
    text = "First sentence. Second sentence! Third sentence? Fourth sentence."
    parts = split_text_into_smaller_parts(text)
    
    assert len(parts) >= 4
    assert all(isinstance(part, str) for part in parts)
    assert "First sentence" in parts[0]

def test_create_frame_narrator(mock_fonts):
    """Test frame creation for narrator."""
    frame = create_frame("Narrator", "Test message", True)
    assert isinstance(frame, np.ndarray)
    assert frame.shape[2] == 3  # Should be a color image
    assert frame.dtype == np.uint8

def test_create_frame_debaters(mock_fonts, mock_avatars):
    """Test frame creation for debaters."""
    # Test Jane's frame
    jane_frame = create_frame("Jane", "Test message", True)
    assert isinstance(jane_frame, np.ndarray)
    assert jane_frame.shape[2] == 3
    
    # Test Valentino's frame
    valentino_frame = create_frame("Valentino", "Test message", True)
    assert isinstance(valentino_frame, np.ndarray)
    assert valentino_frame.shape[2] == 3

@pytest.mark.parametrize("speaker,text,highlighted", [
    ("Narrator", "Test text", True),
    ("Jane", "Test argument", True),
    ("Valentino", "Counter argument", False),
])
def test_create_frame_parameters(mock_fonts, mock_avatars, speaker, text, highlighted):
    """Test frame creation with different parameters."""
    frame = create_frame(speaker, text, highlighted)
    assert isinstance(frame, np.ndarray)
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8

def test_get_segment_audio_file(tmp_path):
    """Test audio file retrieval."""
    # Create mock audio files
    audio_dir = tmp_path / "audio_output"
    audio_dir.mkdir()
    (audio_dir / "part_00.mp3").touch()
    
    with patch('os.listdir', return_value=["part_00.mp3"]):
        with patch('os.path.exists', return_value=True):
            result = get_segment_audio_file(0)
            assert result is not None

def test_get_segment_duration():
    """Test audio duration retrieval."""
    with patch('pydub.AudioSegment') as mock_audio:
        mock_segment = Mock()
        mock_segment.__len__ = Mock(return_value=5000)  # 5 seconds in milliseconds
        mock_audio.from_mp3.return_value = mock_segment
        
        duration = get_segment_duration("test.mp3")
        assert duration == 5.0

def test_get_segment_timing():
    """Test timing information retrieval."""
    mock_timing_data = {
        "segments": [
            {
                "text": "Test text",
                "start_time": 0,
                "end_time": 5
            }
        ]
    }
    
    with patch('builtins.open', mock_open(read_data=json.dumps(mock_timing_data))):
        timing = get_segment_timing("test.mp3")
        assert len(timing) == 1
        assert timing[0]["text"] == "Test text"

def test_create_debate_video(mock_fonts):
    """Test video creation process."""
    mock_segments = [
        {"speaker": "Narrator", "text": "Welcome"},
        {"speaker": "Jane", "text": "Argument"}
    ]
    
    # Create base video frame
    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Setup video clip mock with proper return values
    mock_clip = Mock()
    mock_clip.write_videofile = Mock()
    mock_clip.set_audio = Mock(return_value=mock_clip)
    mock_clip.set_duration = Mock(return_value=mock_clip)
    mock_clip.close = Mock()
    mock_clip.resize = Mock(return_value=mock_clip)
    mock_clip.audio = None
    type(mock_clip).duration = PropertyMock(return_value=5.0)
    
    # Create pool context mock that properly returns frames
    class MockPoolContext:
        def __init__(self):
            self.map = Mock(return_value=[mock_frame] * 150)  # Ensure enough frames
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            return None
        def close(self):
            pass
        def join(self):
            pass

    def mock_get_frame(*args):
        return mock_frame

    timing_data = {'segments': [{'text': 'Test', 'start_time': 0, 'end_time': 5}]}
    
    # Mock pydub AudioSegment for duration calculation
    mock_audio_segment = Mock()
    mock_audio_segment.__len__ = Mock(return_value=5000)  # 5 seconds in milliseconds
    
    def mock_path_join(*args):
        return '/'.join(str(arg) for arg in args)
    
    with patch('debate_to_video.parse_debate_file', return_value=mock_segments), \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs'), \
         patch('multiprocessing.Pool', return_value=MockPoolContext()), \
         patch('moviepy.editor.ImageSequenceClip', return_value=mock_clip), \
         patch('moviepy.editor.AudioFileClip', return_value=mock_clip), \
         patch('moviepy.editor.concatenate_videoclips', return_value=mock_clip), \
         patch('moviepy.editor.VideoFileClip', return_value=mock_clip), \
         patch('pydub.AudioSegment.from_mp3', return_value=mock_audio_segment), \
         patch('os.path.join', side_effect=mock_path_join), \
         patch('os.path.getsize', return_value=1024), \
         patch('os.path.dirname', return_value='outputs'), \
         patch('os.listdir', return_value=['part_00.mp3', 'part_01.mp3']), \
         patch('builtins.open', mock_open(read_data=json.dumps(timing_data))), \
         patch('cv2.imread', return_value=mock_frame), \
         patch('cv2.imwrite', return_value=True), \
         patch('debate_to_video.create_frame', side_effect=mock_get_frame), \
         patch('debate_to_video.get_segment_duration', return_value=5.0):
        
        # Run function
        create_debate_video()
        
        # Verify that video was written
        mock_clip.write_videofile.assert_called_with('outputs/debate.mp4', 
                                                   fps=30, 
                                                   codec='libx264', 
                                                   audio_codec='aac')

def test_error_handling():
    """Test error handling in various functions."""
    # Test with non-existent audio file
    duration = get_segment_duration(None)
    assert duration == 5.0  # Should return default duration
    
    # Test with invalid timing file
    timing = get_segment_timing(None)
    assert timing == []  # Should return empty list
    
    # Test frame creation with invalid parameters
    frame = create_frame(None, None, False)
    assert frame is not None  # Should handle invalid input gracefully
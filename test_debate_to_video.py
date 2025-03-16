import pytest
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Mock the moviepy.editor module before it's imported
sys.modules['moviepy'] = MagicMock()
sys.modules['moviepy.editor'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['PIL.ImageDraw'] = MagicMock()
sys.modules['PIL.ImageFont'] = MagicMock()

# Now we can import from debate_to_video
from debate_to_video import (
    parse_debate_file, get_segment_audio_file, 
    get_segment_duration, create_debate_video
)

@pytest.fixture
def sample_debate_file(tmp_path):
    debate_content = """Ground Statement: AI is beneficial for humanity.
AI Debater 1: Oh my goodness, I feel so strongly about this! While AI has great potential, we need to consider the emotional impact.
AI Debater 2: According to recent studies, AI has demonstrated a 45% increase in productivity across various sectors."""
    
    # Create outputs directory
    os.makedirs(os.path.join(tmp_path, "outputs"), exist_ok=True)
    with open(os.path.join(tmp_path, "outputs/debate.txt"), 'w', encoding='utf-8') as f:
        f.write(debate_content)
    
    # Create audio_output directory
    audio_output = os.path.join(tmp_path, "audio_output")
    os.makedirs(audio_output, exist_ok=True)
    
    # Create mock audio files
    for i in range(3):
        with open(os.path.join(audio_output, f"part_{i}.mp3"), 'wb') as f:
            f.write(b'dummy audio data')
    
    # Create mock assets folder with avatar files
    assets_dir = os.path.join(tmp_path, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    for file in ['jane_avatar.jpg', 'valentino_avatar.jpg']:
        with open(os.path.join(assets_dir, file), 'wb') as f:
            f.write(b'dummy image data')
    
    # Store and change directories
    original_path = os.getcwd()
    os.chdir(tmp_path)
    
    yield tmp_path
    
    # Change back to original directory
    os.chdir(original_path)

@patch('os.path.exists')
@patch('builtins.open')
def test_parse_debate_file(mock_open, mock_exists):
    # Set up exists check for debate.txt
    mock_exists.return_value = True
    
    # Create mock file content
    mock_lines = [
        "Ground Statement: AI is beneficial for humanity.",
        "AI Debater 1: Oh my goodness, I feel so strongly about this! While AI has great potential, we need to consider the emotional impact.",
        "AI Debater 2: According to recent studies, AI has demonstrated a 45% increase in productivity across various sectors."
    ]
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.__iter__.side_effect = [iter(mock_lines)]  # This ensures iterator is created each time
    mock_open.return_value = mock_file
    
    # Call the function
    segments = parse_debate_file()
        
    # Check that we get the expected segments
    assert len(segments) == 3
    assert segments[0]["speaker"] == "Ground"
    assert segments[1]["speaker"] == "Jane"
    assert segments[2]["speaker"] == "Valentino"
    assert "beneficial" in segments[0]["text"]
    assert "emotional impact" in segments[1]["text"]
    assert "productivity" in segments[2]["text"]
    
    # Verify the file was opened correctly
    mock_open.assert_called_once_with("outputs/debate.txt", "r", encoding="utf-8")

@patch('os.listdir')
def test_get_segment_audio_file(mock_listdir, sample_debate_file):
    mock_listdir.return_value = ["part_0.mp3", "part_1.mp3", "part_2.mp3"]
    
    # Test with a valid index
    audio_file = get_segment_audio_file(0)
    assert audio_file is not None
    assert "part_0.mp3" in audio_file
    
    # Test with an invalid index
    audio_file = get_segment_audio_file(100)
    assert audio_file is None

@patch('os.path.exists')
@patch('pydub.AudioSegment.from_mp3')
def test_get_segment_duration(mock_audio, mock_exists):
    # Test with non-existent file
    mock_exists.return_value = False
    duration = get_segment_duration("nonexistent.mp3")
    assert duration == 5.0  # Default duration
    
    # Test with valid audio file
    mock_exists.return_value = True
    mock_audio_instance = MagicMock()
    mock_audio_instance.__len__.return_value = 10000  # 10 seconds in ms
    mock_audio.return_value = mock_audio_instance
    
    duration = get_segment_duration("valid_audio.mp3")
    assert duration == 10.0

@patch('builtins.print')
@patch('debate_to_video.datetime')
@patch('os.makedirs')
@patch('os.path.exists')
@patch('os.listdir')
@patch('os.remove')
@patch('os.rmdir')
@patch('cv2.imread')
@patch('cv2.imwrite')
@patch('cv2.cvtColor')
@patch('PIL.Image.fromarray')
@patch('PIL.ImageDraw.Draw')
@patch('moviepy.editor.ImageClip')
@patch('moviepy.editor.AudioFileClip')
@patch('moviepy.editor.concatenate_videoclips')
def test_create_debate_video(
    mock_concat, mock_audio_clip, mock_image_clip,
    mock_draw, mock_fromarray, mock_cvtcolor,
    mock_imwrite, mock_imread,
    mock_rmdir, mock_remove, mock_listdir,
    mock_exists, mock_makedirs, mock_datetime, mock_print
):
    # Setup datetime mock to return a fixed timestamp
    mock_datetime_instance = Mock()
    mock_datetime_instance.strftime.return_value = "20230101_120000"
    mock_datetime.now.return_value = mock_datetime_instance
    
    # Setup debate segments
    segments = [
        {"speaker": "Ground", "text": "Test ground statement"},
        {"speaker": "Jane", "text": "Test Jane response"},
        {"speaker": "Valentino", "text": "Test Valentino response"}
    ]
    
    # Setup mock frames and image processing
    mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = mock_frame
    mock_imwrite.return_value = True
    
    # Mock PIL image operations
    mock_pil_image = MagicMock()
    mock_draw_obj = MagicMock()
    mock_fromarray.return_value = mock_pil_image
    mock_draw.return_value = mock_draw_obj
    mock_cvtcolor.return_value = mock_frame
    
    # Create a mock font class that returns actual integers
    class MockFont:
        def getlength(self, text):
            # Return a length that's smaller than max_width for the test to pass
            return len(text) * 10  # 10 pixels per character
        
        def getsize(self, text):
            # Return width and height as actual integers
            return (len(text) * 10, 30)
        
        def getbbox(self, text):
            # Return actual integers for the bounding box
            width = len(text) * 10
            return (0, 0, width, 30)
    
    # Create an instance of our MockFont
    mock_font = MockFont()
    
    # Patch ImageFont.truetype to return our mock font
    with patch('PIL.ImageFont.truetype', return_value=mock_font):
        # Setup directory listing mock to handle both audio and frame paths
        def mock_listdir_impl(path):
            if path == 'audio_output':
                return [f"part_{i}.mp3" for i in range(3)]
            elif path == 'temp_frames':
                return [f"frame_{i:03d}.png" for i in range(3)]
            return []
        
        mock_listdir.side_effect = mock_listdir_impl
        
        # Setup video components
        mock_clip = MagicMock()
        mock_clip.write_videofile.return_value = None
        mock_clip.set_duration.return_value = mock_clip
        mock_clip.set_audio.return_value = mock_clip
        mock_image_clip.return_value = mock_clip
        mock_concat.return_value = mock_clip
        
        # Mock audio clip with duration
        mock_audio = MagicMock()
        mock_audio.duration = 5.0
        mock_audio_clip.return_value = mock_audio
        
        # Set file existence check to True for all paths
        mock_exists.return_value = True
        
        # Create temporary directory for test
        with patch('debate_to_video.parse_debate_file', return_value=segments):
            # Call the function under test
            create_debate_video()
        
        # Verify the function's behavior
        mock_makedirs.assert_called_with("temp_frames", exist_ok=True)
        
        # Verify frames were created and saved
        frame_paths = [f"temp_frames/frame_{i:03d}.png" for i in range(len(segments))]
        for frame_path in frame_paths:
            mock_imwrite.assert_any_call(frame_path, mock_frame)
        
        # Verify video processing
        assert mock_image_clip.call_count == len(segments)
        assert mock_clip.set_duration.call_count == len(segments)
        
        # Verify final video creation
        mock_concat.assert_called_once()
        mock_clip.write_videofile.assert_called_once_with(
            "debate_20230101_120000.mp4",
            fps=24,
            codec='libx264',
            audio_codec='aac'
        )
        
        # Verify cleanup
        assert mock_remove.call_count == len(frame_paths)
        mock_rmdir.assert_called_once_with("temp_frames")

def test_parse_debate_file_with_narrator():
    """Test parsing debate file with narrator introduction."""
    mock_lines = [
        "Narrator: Welcome to our AI debate. In this video, two AI debaters will engage...",
        "Ground Statement: AI is beneficial for humanity.",
        "AI Debater 1: First response",
        "AI Debater 2: Second response"
    ]
    
    with patch('builtins.open', mock_open(read_data="\n".join(mock_lines))):
        segments = parse_debate_file()
        
        assert len(segments) == 4
        assert segments[0]["speaker"] == "Narrator"
        assert "Welcome to our AI debate" in segments[0]["text"]
        assert segments[1]["speaker"] == "Ground"

@patch('cv2.imread')
@patch('cv2.imwrite')
@patch('PIL.Image.fromarray')
@patch('PIL.ImageDraw.Draw')
def test_create_frame_narrator(mock_draw, mock_fromarray, mock_imwrite, mock_imread):
    """Test frame creation for narrator text."""
    # Setup mocks
    mock_frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    mock_pil_image = MagicMock()
    mock_draw_obj = MagicMock()
    mock_fromarray.return_value = mock_pil_image
    mock_draw.return_value = mock_draw_obj
    
    # Create a mock font
    class MockFont:
        def getlength(self, text):
            return len(text) * 10
        def getsize(self, text):
            return (len(text) * 10, 30)
        def getbbox(self, text):
            width = len(text) * 10
            return (0, 0, width, 30)
    
    mock_font = MockFont()
    
    with patch('PIL.ImageFont.truetype', return_value=mock_font):
        # Test narrator frame creation
        frame = create_frame(
            speaker="Narrator",
            text="Welcome to our AI debate",
            highlighted=False,
            current_time=0,
            total_duration=5.0
        )
        
        # Verify frame was created
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3)

def test_split_text_into_chunks():
    """Test text chunking functionality."""
    # Test short text
    short_text = "This is a short text"
    chunks = split_text_into_chunks(short_text, chunk_size=5)
    assert len(chunks) == 1
    assert chunks[0] == short_text
    
    # Test long text
    long_text = "This is a much longer text that should be split into multiple chunks for better readability"
    chunks = split_text_into_chunks(long_text, chunk_size=5)
    assert len(chunks) > 1
    for chunk in chunks:
        words = chunk.split()
        assert len(words) <= 5

def test_create_frame_with_different_timings():
    """Test frame creation with different timing positions."""
    text = "This is a long text that should be split into multiple chunks"
    duration = 10.0
    
    with patch('cv2.imread'), patch('cv2.imwrite'), \
         patch('PIL.Image.fromarray'), patch('PIL.ImageDraw.Draw'):
        
        # Test frame at start
        frame1 = create_frame("Jane", text, True, current_time=0, total_duration=duration)
        
        # Test frame at middle
        frame2 = create_frame("Jane", text, True, current_time=5.0, total_duration=duration)
        
        # Test frame at end
        frame3 = create_frame("Jane", text, True, current_time=9.9, total_duration=duration)
        
        assert frame1 is not None
        assert frame2 is not None
        assert frame3 is not None
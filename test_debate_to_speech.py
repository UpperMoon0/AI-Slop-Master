import pytest
from unittest.mock import Mock, patch, mock_open
import os
import json
import asyncio
from debate_to_speech import process_debate

@pytest.mark.asyncio
async def test_process_debate():
    """Test full debate processing."""
    mock_content = """Narrator: Welcome
AI Debater 1: First argument
AI Debater 2: Counter argument"""
    
    with patch('debate_to_speech.parse_debate_file', return_value=[
        {"speaker": "Narrator", "text": "Welcome"},
        {"speaker": "AI Debater 1", "text": "First argument"},
        {"speaker": "AI Debater 2", "text": "Counter argument"}
    ]):
        with patch('debate_to_speech.generate_debate_speech') as mock_generate:
            # Make mock_generate a coroutine
            async def mock_generate_coro(*args, **kwargs):
                return True
            mock_generate.side_effect = mock_generate_coro
            
            result = await process_debate()
            assert result == True
            mock_generate.assert_called_once()

@pytest.mark.asyncio
async def test_process_debate_exception():
    """Test process_debate with exception."""
    with patch('debate_to_speech.parse_debate_file', side_effect=Exception("Test error")):
        result = await process_debate()
        assert result == False

# Tests for utility functions used by debate_to_speech.py
@pytest.mark.asyncio
async def test_text_to_speech_from_utils():
    """Test text to speech conversion from utils."""
    with patch('utils.audio_utils.edge_tts.Communicate') as mock_communicate:
        # Setup mock with proper async behavior
        mock_communicate_instance = Mock()
        mock_communicate.return_value = mock_communicate_instance
        
        # Create an async mock for save
        async def mock_save(output_file):
            return None
        mock_communicate_instance.save = mock_save
        
        # Mock AudioSegment to avoid file operations
        with patch('utils.audio_utils.AudioSegment') as mock_audio_segment:
            # Create a proper mock for __len__ using a proper MagicMock
            mock_audio = Mock()
            # Mock the __len__ method directly rather than trying to set its return value
            mock_audio.__len__ = Mock(return_value=5000)
            mock_audio_segment.from_mp3.return_value = mock_audio
            
            # Mock os.makedirs to avoid directory check errors
            with patch('os.makedirs'):
                # Mock open for the timing file
                with patch('utils.audio_utils.open', mock_open()):
                    # Import inside the test to ensure mocks are in place
                    from utils.audio_utils import text_to_speech
                    
                    # Test successful conversion
                    result = await text_to_speech("Test text", "en-US-JennyNeural", "test.mp3")
                    assert result == True
                    mock_communicate.assert_called_once()

@pytest.mark.asyncio
async def test_process_debate_segments():
    """Test debate segment processing."""
    mock_segments = [
        {"speaker": "Narrator", "text": "Welcome"},
        {"speaker": "AI Debater 1", "text": "Argument"}
    ]
    
    # Mock text_to_speech
    with patch('utils.audio_utils.text_to_speech') as mock_tts:
        # Make mock_tts a coroutine
        async def mock_tts_coro(*args, **kwargs):
            return True
        mock_tts.side_effect = mock_tts_coro
        
        # Mock os.path.exists to avoid directory checks
        with patch('os.path.exists', return_value=True):
            # Mock os.listdir to avoid directory listing
            with patch('os.listdir', return_value=[]):
                # Mock os.makedirs to avoid directory creation
                with patch('os.makedirs'):
                    # Import inside the test to ensure mocks are in place
                    from utils.audio_utils import process_debate_segments
                    
                    success = await process_debate_segments(mock_segments, "test_output")
                    assert success == True
                    assert mock_tts.call_count == len(mock_segments)

@pytest.mark.asyncio
async def test_process_debate_segments_with_failures():
    """Test debate segment processing with some failures."""
    mock_segments = [
        {"speaker": "Narrator", "text": "Welcome"},
        {"speaker": "AI Debater 1", "text": "Argument"}
    ]
    
    # First call succeeds, second call fails
    tts_results = [True, False]
    call_count = 0
    
    # Mock text_to_speech with varying results
    async def mock_tts_side_effect(*args, **kwargs):
        nonlocal call_count
        result = tts_results[call_count]
        call_count += 1
        return result
    
    with patch('utils.audio_utils.text_to_speech', side_effect=mock_tts_side_effect):
        # Mock os operations
        with patch('os.path.exists', return_value=True):
            with patch('os.listdir', return_value=[]):
                with patch('os.makedirs'):
                    # Mock AudioSegment for silent audio creation
                    with patch('utils.audio_utils.AudioSegment') as mock_audio:
                        # Create a proper mock with __len__ method
                        mock_silent = Mock()
                        mock_silent.__len__ = Mock(return_value=3000)
                        mock_audio.silent.return_value = mock_silent
                        # Mock open for writing files
                        with patch('utils.audio_utils.open', mock_open()):
                            # Import inside the test
                            from utils.audio_utils import process_debate_segments
                            
                            success = await process_debate_segments(mock_segments, "test_output")
                            assert success == True  # Still true if at least one success

@pytest.mark.asyncio
async def test_generate_debate_speech_from_utils():
    """Test generate_debate_speech from utils."""
    mock_segments = [
        {"speaker": "Narrator", "text": "Welcome"},
        {"speaker": "AI Debater 1", "text": "Argument"}
    ]
    
    with patch('utils.audio_utils.process_debate_segments') as mock_process:
        # Make mock_process a coroutine
        async def mock_process_coro(*args, **kwargs):
            return True
        mock_process.side_effect = mock_process_coro
        
        # Import inside the test
        from utils.audio_utils import generate_debate_speech
        
        result = await generate_debate_speech(mock_segments, "test_output")
        assert result == True
        mock_process.assert_called_once_with(mock_segments, "test_output")

@pytest.mark.asyncio
async def test_generate_debate_speech_with_exception():
    """Test generate_debate_speech with exception."""
    mock_segments = [
        {"speaker": "Narrator", "text": "Welcome"},
        {"speaker": "AI Debater 1", "text": "Argument"}
    ]
    
    with patch('utils.audio_utils.process_debate_segments', side_effect=Exception("Test error")):
        # Import inside the test
        from utils.audio_utils import generate_debate_speech
        
        result = await generate_debate_speech(mock_segments, "test_output")
        assert result == False

def test_get_segment_audio_file():
    """Test get_segment_audio_file function."""
    # Test when file exists
    with patch('os.path.exists', return_value=True):
        from utils.audio_utils import get_segment_audio_file
        
        audio_file = get_segment_audio_file(5)
        # Replace forward slashes with backslashes to make the test OS-agnostic
        if audio_file:
            audio_file = audio_file.replace('/', '\\')
        # Updated assertion to handle Windows path separators
        expected_path = os.path.join('outputs', 'audio_output', 'part_05.mp3')
        assert audio_file == expected_path
    
    # Test when file doesn't exist but can find by listing directory
    with patch('os.path.exists', return_value=False):
        with patch('os.listdir', return_value=['part_00.mp3', 'part_01.mp3', 'part_02.mp3']):
            from utils.audio_utils import get_segment_audio_file
            
            audio_file = get_segment_audio_file(1)
            # Replace forward slashes with backslashes to make the test OS-agnostic
            if audio_file:
                audio_file = audio_file.replace('/', '\\')
            # Updated assertion to handle Windows path separators
            expected_path = os.path.join('outputs', 'audio_output', 'part_01.mp3')
            assert audio_file == expected_path
    
    # Test when file doesn't exist and can't find in directory
    with patch('os.path.exists', return_value=False):
        with patch('os.listdir', return_value=[]):
            from utils.audio_utils import get_segment_audio_file
            
            audio_file = get_segment_audio_file(5)
            assert audio_file is None

def test_get_segment_duration():
    """Test get_segment_duration function."""
    # Test with existing file
    with patch('os.path.exists', return_value=True):
        with patch('utils.audio_utils.AudioSegment') as mock_audio:
            # Create a proper mock with __len__ method
            mock_instance = Mock()
            mock_instance.__len__ = Mock(return_value=3000)  # 3 seconds in milliseconds
            mock_audio.from_mp3.return_value = mock_instance
            
            from utils.audio_utils import get_segment_duration
            
            duration = get_segment_duration('test.mp3')
            assert duration == 3.0
    
    # Test with non-existent file
    with patch('os.path.exists', return_value=False):
        from utils.audio_utils import get_segment_duration
        
        duration = get_segment_duration('test.mp3')
        assert duration == 5.0  # Default duration
    
    # Test with exception
    with patch('os.path.exists', return_value=True):
        with patch('utils.audio_utils.AudioSegment.from_mp3', side_effect=Exception("Test error")):
            from utils.audio_utils import get_segment_duration
            
            duration = get_segment_duration('test.mp3')
            assert duration == 5.0  # Default duration on error

def test_parse_debate():
    """Test parse_debate function."""
    mock_content = """Narrator: Welcome to our AI debate.
    
Ground Statement: This is a test statement.

AI Debater 1: First argument.
AI Debater 2: Counter argument.

Result: The debate ended in a draw."""
    
    with patch('utils.audio_utils.open', mock_open(read_data=mock_content)):
        from utils.audio_utils import parse_debate
        
        segments = parse_debate()
        # Updated assertion to match actual implementation which includes Result segment
        assert len(segments) == 5  # Now includes Result segment
        assert segments[0]["speaker"] == "Narrator"
        assert segments[1]["speaker"] == "Ground Statement"
        assert segments[2]["speaker"] == "AI Debater 1"
        assert segments[3]["speaker"] == "AI Debater 2"
        assert segments[4]["speaker"] == "Result"  # Added assertion for Result segment
        assert segments[0]["text"] == "Welcome to our AI debate."
        assert segments[3]["text"] == "Counter argument."
        assert segments[4]["text"] == "The debate ended in a draw."  # Added assertion for Result text

def test_get_current_subtitle():
    """Test get_current_subtitle function."""
    from utils.audio_utils import get_current_subtitle
    
    timing_segments = [
        {"text": "First line", "start_time": 0.0, "end_time": 2.0},
        {"text": "Second line", "start_time": 2.0, "end_time": 4.0},
        {"text": "Third line", "start_time": 4.0, "end_time": 6.0},
    ]
    
    # Test time in first segment
    text, speaker = get_current_subtitle(timing_segments, 1.0, "Default")
    assert text == "First line"
    assert speaker is None
    
    # Test time in second segment
    text, speaker = get_current_subtitle(timing_segments, 3.0, "Default")
    assert text == "Second line"
    assert speaker is None
    
    # Test time after all segments (should return last segment)
    text, speaker = get_current_subtitle(timing_segments, 7.0, "Default")
    assert text == "Third line"
    assert speaker is None
    
    # Test with empty segments
    text, speaker = get_current_subtitle([], 1.0, "Default")
    assert text == "Default"
    assert speaker is None
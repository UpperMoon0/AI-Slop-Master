import pytest
from unittest.mock import Mock, patch, mock_open
import os
import json
import asyncio
from debate_to_speech import text_to_speech, process_debate_segments, generate_debate_speech, process_debate

@pytest.mark.asyncio
async def test_text_to_speech():
    """Test text to speech conversion."""
    with patch('edge_tts.Communicate') as mock_communicate:
        # Setup mock with proper async behavior
        mock_communicate_instance = Mock()
        mock_communicate.return_value = mock_communicate_instance
        
        # Create an async mock for save
        async def mock_save(output_file):
            return None
        mock_communicate_instance.save = mock_save
        
        # Test successful conversion
        result = await text_to_speech("Test text", "en-US-JennyNeural", "test.mp3")
        assert result == True
        mock_communicate.assert_called_once()

@pytest.mark.asyncio
async def test_process_debate_segments():
    """Test debate segment processing."""
    mock_segments = [
        {"speaker": "Narrator", "text": "Welcome"},
        {"speaker": "Jane", "text": "Argument"}
    ]
    
    with patch('debate_to_speech.text_to_speech') as mock_tts:
        # Make mock_tts a coroutine
        async def mock_tts_coro(*args, **kwargs):
            return True
        mock_tts.side_effect = mock_tts_coro
        
        success = await process_debate_segments(mock_segments)
        assert success == True
        assert mock_tts.call_count == len(mock_segments)

@pytest.mark.asyncio
async def test_generate_debate_speech():
    """Test debate speech generation."""
    mock_segments = [
        {"speaker": "Narrator", "text": "Welcome"},
        {"speaker": "Jane", "text": "Argument"}
    ]
    
    with patch('debate_to_speech.process_debate_segments') as mock_process:
        # Make mock_process a coroutine
        async def mock_process_coro(*args, **kwargs):
            return True
        mock_process.side_effect = mock_process_coro
        
        result = await generate_debate_speech(mock_segments)
        assert result == True
        mock_process.assert_called_once()

@pytest.mark.asyncio
async def test_process_debate():
    """Test full debate processing."""
    mock_content = """Narrator: Welcome
AI Debater 1: First argument
AI Debater 2: Counter argument"""
    
    with patch('builtins.open', mock_open(read_data=mock_content)):
        with patch('debate_to_speech.generate_debate_speech') as mock_generate:
            # Make mock_generate a coroutine
            async def mock_generate_coro(*args, **kwargs):
                return True
            mock_generate.side_effect = mock_generate_coro
            
            await process_debate()
            mock_generate.assert_called_once()

@pytest.mark.asyncio
async def test_debate_file_parsing():
    """Test debate file parsing."""
    mock_content = """Narrator: Welcome
AI Debater 1: First argument
AI Debater 2: Counter argument"""
    
    with patch('builtins.open', mock_open(read_data=mock_content)):
        segments = []
        with open('outputs/debate.txt', 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                if ':' in line:
                    speaker, text = line.split(':', 1)
                    segments.append({
                        "speaker": speaker.strip(),
                        "text": text.strip()
                    })
        
        assert len(segments) == 3
        assert segments[0]["speaker"] == "Narrator"
        assert segments[1]["speaker"] == "AI Debater 1"
        assert segments[2]["speaker"] == "AI Debater 2"
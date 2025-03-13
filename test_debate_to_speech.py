import pytest
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from debate_to_speech import process_debate, text_to_speech, VOICE_NARRATOR, VOICE_AI1, VOICE_AI2

@pytest.fixture
def sample_debate_file(tmp_path):
    debate_content = """Ground Statement: AI is beneficial for humanity.
AI Debater 1: Oh my goodness, I feel so strongly about this! While AI has great potential, we need to consider the emotional impact on people.
AI Debater 2: According to recent studies, AI has demonstrated a 45% increase in productivity across various sectors."""
    
    debate_file = tmp_path / "debate.txt"
    debate_file.write_text(debate_content)
    
    # Store the original path
    original_path = os.getcwd()
    
    # Change to the temporary directory
    os.chdir(tmp_path)
    
    yield tmp_path
    
    # Change back to the original directory
    os.chdir(original_path)

@pytest.mark.asyncio
async def test_text_to_speech():
    mock_communicate = AsyncMock()
    mock_communicate.save = AsyncMock()
    
    with patch('edge_tts.Communicate', return_value=mock_communicate) as mock_tts:
        await text_to_speech("Test text", VOICE_NARRATOR, "test.mp3")
        
        mock_tts.assert_called_once_with("Test text", VOICE_NARRATOR)
        mock_communicate.save.assert_called_once_with("test.mp3")

@pytest.mark.asyncio
async def test_process_debate(sample_debate_file):
    with patch('debate_to_speech.text_to_speech', new_callable=AsyncMock) as mock_tts:
        await process_debate()
        
        # We expect three calls (ground statement, AI 1, AI 2)
        assert mock_tts.call_count == 3
        
        # Verify the voices used for each part
        calls = mock_tts.call_args_list
        assert calls[0].args[1] == VOICE_NARRATOR  # Ground statement
        assert calls[1].args[1] == VOICE_AI1      # AI Debater 1
        assert calls[2].args[1] == VOICE_AI2      # AI Debater 2
        
        # Verify audio_output directory was created
        assert os.path.exists('audio_output')
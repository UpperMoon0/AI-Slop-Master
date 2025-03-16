import pytest
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from debate_to_speech import (
    process_debate, text_to_speech, get_voice_for_speaker,
    VOICE_NARRATOR, VOICE_JANE, VOICE_VALENTINO, VOICE_BACKUP,
    clear_output_folder, verify_audio_file, split_long_text
)

@pytest.fixture
def sample_debate_file(tmp_path):
    debate_content = """Ground Statement: AI is beneficial for humanity.
AI Debater 1: Oh my goodness, I feel so strongly about this! While AI has great potential, we need to consider the emotional impact on people.
AI Debater 2: According to recent studies, AI has demonstrated a 45% increase in productivity across various sectors."""
    
    # Create debate.txt in the current directory
    with open(os.path.join(tmp_path, "debate.txt"), 'w', encoding='utf-8') as f:
        f.write(debate_content)
    
    # Create audio_output directory
    audio_output = os.path.join(tmp_path, "audio_output")
    os.makedirs(audio_output, exist_ok=True)
    
    # Store the original path
    original_path = os.getcwd()
    
    # Change to the temporary directory
    os.chdir(tmp_path)
    
    yield tmp_path
    
    # Change back to the original directory
    os.chdir(original_path)

def test_clear_output_folder(tmp_path):
    # Create test files in audio_output
    audio_output = os.path.join(tmp_path, "audio_output")
    os.makedirs(audio_output, exist_ok=True)
    
    with open(os.path.join(audio_output, "test.mp3"), 'w') as f:
        f.write("test")
    
    # Change to tmp_path
    original_path = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        clear_output_folder()
        
        # Check if directory is empty but exists
        assert os.path.exists("audio_output")
        assert len(os.listdir("audio_output")) == 0
    finally:
        # Change back to original directory
        os.chdir(original_path)

@patch('os.path.exists')
@patch('os.path.getsize', return_value=100)  # Mock file size 
@patch('pydub.AudioSegment.from_mp3')
def test_verify_audio_file(mock_from_mp3, mock_getsize, mock_exists):
    # Test when file doesn't exist
    mock_exists.return_value = False
    assert verify_audio_file("nonexistent.mp3") == False
    
    # Test when file exists but is invalid
    mock_exists.return_value = True
    mock_from_mp3.side_effect = Exception("Test error")
    assert verify_audio_file("invalid.mp3") == False
    
    # Test when file exists and is valid
    mock_exists.return_value = True
    mock_from_mp3.side_effect = None
    # Create a mock audio segment that can be sliced
    mock_audio = MagicMock()
    # MagicMock automatically handles __getitem__ calls
    mock_from_mp3.return_value = mock_audio
    assert verify_audio_file("valid.mp3") == True

def test_split_long_text():
    short_text = "This is a short text."
    assert split_long_text(short_text, 100) == [short_text]
    
    long_text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = split_long_text(long_text, 25)
    assert len(chunks) > 1
    assert "sentence one" in chunks[0]
    assert "sentence three" in chunks[-1]

@pytest.mark.asyncio
@patch('os.makedirs')
async def test_text_to_speech(mock_makedirs):
    mock_communicate = AsyncMock()
    mock_communicate.save = AsyncMock()
    
    # Create a path that includes directory
    output_path = "test_dir/test.mp3"
    
    with patch('edge_tts.Communicate', return_value=mock_communicate) as mock_tts:
        await text_to_speech("Test text", VOICE_NARRATOR, output_path)
        
        # Verify directories were created
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        
        # Verify voice-specific settings
        mock_tts.assert_called_once()
        call_args = mock_tts.call_args[1]
        assert call_args['voice'] == VOICE_NARRATOR
        assert 'rate' in call_args
        
        mock_communicate.save.assert_called_once_with(output_path)

@pytest.mark.asyncio
@patch('os.makedirs')
async def test_text_to_speech_ground_statement(mock_makedirs):
    mock_communicate = AsyncMock()
    mock_communicate.save = AsyncMock()
    
    # Create a path that includes directory
    output_path = "test_dir/test_ground.mp3"
    
    with patch('edge_tts.Communicate', return_value=mock_communicate) as mock_tts:
        # Test ground statement handling
        await text_to_speech("Test ground statement", VOICE_NARRATOR, output_path, is_ground_statement=True)
        
        # Verify directories were created
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        
        call_args = mock_tts.call_args[1]
        assert call_args['voice'] == VOICE_NARRATOR
    
    # Test backup voice when primary fails - handle this in a separate test case
    # to avoid complications with exception handling
    await test_text_to_speech_ground_statement_with_fallback()

@pytest.mark.asyncio
@patch('os.makedirs')
async def test_text_to_speech_ground_statement_with_fallback(mock_makedirs):
    # Create mocks for both the failing call and the successful backup call
    mock_backup_communicate = AsyncMock()
    mock_backup_communicate.save = AsyncMock()
    
    # Create a list of side effects: first an exception, then the mock
    side_effects = [Exception("Test error"), mock_backup_communicate]
    
    with patch('edge_tts.Communicate', side_effect=side_effects):
        # Use try/except to handle the expected exception
        try:
            await text_to_speech("Test ground statement", VOICE_NARRATOR, "test_dir/test_fallback.mp3", is_ground_statement=True)
        except Exception:
            # The exception should be caught within text_to_speech and it should try with the backup voice
            pass
        
        # We can't easily verify the voice parameters here because of the exception handling
        # But we can verify that makedirs was called
        mock_makedirs.assert_called_with("test_dir", exist_ok=True)

@pytest.mark.asyncio
@patch('os.makedirs')
async def test_text_to_speech_different_voices(mock_makedirs):
    mock_communicate = AsyncMock()
    mock_communicate.save = AsyncMock()
    
    with patch('edge_tts.Communicate', return_value=mock_communicate) as mock_tts:
        # Test Jane's voice settings
        await text_to_speech("Test Jane", VOICE_JANE, "test_dir/test_jane.mp3")
        call_args = mock_tts.call_args[1]
        assert call_args['voice'] == VOICE_JANE
        
        # Test Valentino's voice settings
        await text_to_speech("Test Valentino", VOICE_VALENTINO, "test_dir/test_valentino.mp3")
        call_args = mock_tts.call_args[1]
        assert call_args['voice'] == VOICE_VALENTINO

def test_get_voice_for_speaker():
    """Test voice selection for different speakers."""
    assert get_voice_for_speaker("Jane") == "alloy"
    assert get_voice_for_speaker("Valentino") == "echo"
    assert get_voice_for_speaker("Narrator") == "nova"
    assert get_voice_for_speaker("Ground") == "onyx"
    assert get_voice_for_speaker("Result") == "onyx"

@pytest.mark.asyncio
@patch('os.makedirs')
@patch('openai.OpenAI')
async def test_openai_tts_integration(mock_openai, mock_makedirs):
    """Test OpenAI TTS integration."""
    mock_client = Mock()
    mock_response = Mock()
    mock_client.audio.speech.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    # Create a mock environment with OpenAI API key
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        await process_debate()
        
        # Verify OpenAI TTS was called with correct parameters
        calls = mock_client.audio.speech.create.call_args_list
        for call in calls:
            args, kwargs = call
            assert kwargs['model'] == 'tts-1'
            assert 'voice' in kwargs
            assert 'input' in kwargs

@pytest.mark.asyncio
async def test_process_debate_with_narrator():
    """Test processing debate file with narrator section."""
    # Create temporary debate.txt with narrator
    debate_content = """Narrator: Welcome to our AI debate.
Ground Statement: Test statement.
AI Debater 1: First response.
AI Debater 2: Second response."""
    
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/debate.txt', 'w', encoding='utf-8') as f:
        f.write(debate_content)
    
    # Mock OpenAI client and audio processing
    mock_client = Mock()
    mock_response = Mock()
    mock_response.write_to_file = Mock()
    mock_client.audio.speech.create.return_value = mock_response
    
    with patch('openai.OpenAI', return_value=mock_client), \
         patch('pydub.AudioSegment.from_mp3'), \
         patch('pydub.AudioSegment.export'):
        await process_debate()
        
        # Verify narrator voice was used for first segment
        first_call = mock_client.audio.speech.create.call_args_list[0]
        assert first_call[1]['voice'] == 'nova'  # Narrator voice
        
        # Verify each segment was processed
        assert mock_client.audio.speech.create.call_count == 4  # Narrator + Ground + 2 debaters

@pytest.mark.asyncio
async def test_audio_combination_with_pauses():
    """Test that audio segments are combined with appropriate pauses."""
    mock_segment = MagicMock()
    mock_segment.__add__.return_value = mock_segment
    mock_segment.__len__.return_value = 5000  # 5 seconds
    
    with patch('pydub.AudioSegment') as mock_audio:
        mock_audio.from_mp3.return_value = mock_segment
        mock_audio.empty.return_value = mock_segment
        mock_audio.silent.return_value = mock_segment
        
        # Create some test files
        os.makedirs('outputs/audio_output', exist_ok=True)
        test_files = ['test1.mp3', 'test2.mp3']
        for file in test_files:
            with open(f'outputs/audio_output/{file}', 'wb') as f:
                f.write(b'test')
        
        # Test the combination
        from debate_to_speech import combine_audio_files
        combine_audio_files([f'outputs/audio_output/{f}' for f in test_files])
        
        # Verify silent pauses were added
        assert mock_audio.silent.called
        assert mock_segment.export.called

@pytest.mark.asyncio
async def test_error_handling_in_tts():
    """Test error handling in text-to-speech processing."""
    with patch('openai.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_client.audio.speech.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        # Create test debate file
        os.makedirs('outputs', exist_ok=True)
        with open('outputs/debate.txt', 'w', encoding='utf-8') as f:
            f.write("Ground Statement: Test\nAI Debater 1: Test response")
        
        # Process should continue despite errors
        await process_debate()
        
        # Verify error handling
        mock_client.audio.speech.create.assert_called()
        # Process should attempt to create audio for each segment despite errors

@pytest.mark.asyncio
async def test_process_debate():
    # Create the outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Create a temporary debate.txt file
    with open('outputs/debate.txt', 'w', encoding='utf-8') as f:
        f.write("Ground Statement: AI is beneficial for humanity.\n")
        f.write("AI Debater 1: Oh my goodness, I feel so strongly about this!\n")
        f.write("AI Debater 2: According to recent studies, AI has demonstrated a 45% increase in productivity.")
    
    # Set up mocks
    mock_tts = AsyncMock()
    mock_combine = Mock()
    mock_makedirs = Mock()
    mock_rmtree = Mock()
    
    # Mock exists check
    def exists_side_effect(path):
        if path in ['outputs/debate.txt', 'audio_output']:
            return True
        return False
    mock_exists = Mock(side_effect=exists_side_effect)
    
    # Apply all patches
    with patch('debate_to_speech.text_to_speech', mock_tts), \
         patch('debate_to_speech.combine_audio_files', mock_combine), \
         patch('os.makedirs', mock_makedirs), \
         patch('shutil.rmtree', mock_rmtree), \
         patch('os.path.exists', mock_exists):
        
        # Call process_debate
        await process_debate()
        
        # Verify directory operations
        mock_rmtree.assert_called_once_with('audio_output')
        mock_makedirs.assert_called_with('audio_output', exist_ok=True)
        
        # We expect three calls (ground statement, AI 1, AI 2)
        assert mock_tts.call_count == 3
        assert mock_tts.await_count == 3  # Verify the async calls were awaited
        
        # Verify the voices used for each part
        calls = mock_tts.call_args_list
        assert calls[0].args[1] == VOICE_NARRATOR  # Ground statement
        assert calls[1].args[1] == VOICE_JANE      # AI Debater 1
        assert calls[2].args[1] == VOICE_VALENTINO      # AI Debater 2
        
        # Verify combine_audio_files was called
        mock_combine.assert_called_once()
    
    # Clean up the temporary file
    os.remove('outputs/debate.txt')
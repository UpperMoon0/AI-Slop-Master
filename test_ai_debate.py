import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Mock external dependencies to avoid import errors
sys.modules['moviepy'] = MagicMock()
sys.modules['moviepy.editor'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['PIL.ImageDraw'] = MagicMock()
sys.modules['PIL.ImageFont'] = MagicMock()

from collections import deque

# Import the AIDebater class but patch the imports it uses
with patch('ai_debate.process_debate'), patch('ai_debate.create_debate_video'):
    from ai_debate import AIDebater

@pytest.fixture
def mock_openai_response():
    return Mock(choices=[Mock(message=Mock(content="Test response"))])

@pytest.fixture
def debater():
    return AIDebater()

def test_initialize_debater():
    debater = AIDebater()
    assert isinstance(debater.debate_history, deque)
    assert len(debater.debate_history) == 0

@patch('openai.OpenAI')
def test_generate_response(mock_openai, debater, mock_openai_response):
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    mock_openai.return_value = mock_client
    
    debater.client = mock_client
    response = debater.generate_response("test prompt", "AI 1")
    
    assert response == "Test response"
    assert len(debater.debate_history) == 1
    assert debater.debate_history[0] == "Test response"

@patch('openai.OpenAI')
@patch('ai_debate.AIDebater._check_similarity', return_value=True)  # Force surrender by making responses seem similar
def test_debate_flow(mock_check_similarity, mock_openai, debater, mock_openai_response):
    mock_client = Mock()
    # Make only two responses (one for each AI) before surrender
    responses = [
        mock_openai_response,  # AI 1 first response
        mock_openai_response,  # AI 2 first response
        Mock(choices=[Mock(message=Mock(content="surrender"))])  # AI 1 surrenders
    ]
    mock_client.chat.completions.create.side_effect = responses
    mock_openai.return_value = mock_client
    
    debater.client = mock_client
    ground_statement = "Test statement"
    results = debater.debate(ground_statement, generate_audio=False)
    
    assert results[0] == ground_statement
    assert len(results) == 4  # Ground statement + 2 responses + surrender
    assert mock_client.chat.completions.create.call_count == 3

@patch('openai.OpenAI')
def test_surrender_condition(mock_openai, debater):
    mock_client = Mock()
    surrender_response = Mock(choices=[Mock(message=Mock(content="I surrender"))])
    mock_client.chat.completions.create.return_value = surrender_response
    mock_openai.return_value = mock_client
    
    debater.client = mock_client
    results = debater.debate("Test statement", generate_audio=False)
    
    assert len(results) == 2  # Ground statement + surrender response
    assert "surrender" in results[-1].lower()

@patch('openai.OpenAI')
@patch('os.path.exists')
def test_debate_file_output(mock_exists, mock_openai, debater, mock_openai_response):
    # Make os.path.exists return True when checking for debate.txt
    mock_exists.return_value = True
    
    mock_client = Mock()
    # Make only two responses (one for each AI) before surrender
    responses = [
        mock_openai_response,  # AI 1 first response
        mock_openai_response,  # AI 2 first response
        Mock(choices=[Mock(message=Mock(content="surrender"))])  # AI 1 surrenders
    ]
    mock_client.chat.completions.create.side_effect = responses
    mock_openai.return_value = mock_client
    
    debater.client = mock_client
    ground_statement = "Test statement"
    
    # Mock file operations
    mock_open = MagicMock()
    with patch('builtins.open', mock_open):
        results = debater.debate(ground_statement, generate_audio=False)
    
    # Verify that open was called for writing the output file
    mock_open.assert_called()
    assert mock_client.chat.completions.create.call_count == 3

@patch('openai.OpenAI')
def test_debate_history_in_context(mock_openai, debater):
    mock_client = Mock()
    responses = [
        Mock(choices=[Mock(message=Mock(content="First response"))]),
        Mock(choices=[Mock(message=Mock(content="Second response"))]),
        Mock(choices=[Mock(message=Mock(content="surrender"))])
    ]
    mock_client.chat.completions.create.side_effect = responses
    mock_openai.return_value = mock_client
    
    debater.client = mock_client
    ground_statement = "Test statement"
    
    # Use a mock for file operations
    with patch('builtins.open', MagicMock()):
        debater.debate(ground_statement, generate_audio=False)
    
    # Get the calls made to the API
    calls = mock_client.chat.completions.create.call_args_list
    
    # Check second API call includes previous context
    second_call = calls[1][1]['messages'][0]['content']
    assert "Ground Statement: Test statement" in second_call
    assert "Jane: First response" in second_call

    # Verify both AIs get the complete history
    first_call = calls[0][1]['messages'][0]['content']
    assert "Ground Statement: Test statement" in first_call

@patch('openai.OpenAI')
def test_surrender_instructions_included(mock_openai, debater):
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    )
    mock_openai.return_value = mock_client
    
    debater.client = mock_client
    debater.generate_response("test prompt", "AI Debater 1")
    
    # Check if surrender instructions are in the system prompt
    system_message = mock_client.chat.completions.create.call_args[1]['messages'][0]['content']
    assert "surrender" in system_message.lower()
    assert "If you find your position difficult to defend" in system_message

def test_ai_personality_differences():
    debater = AIDebater()
    
    # Test Jane's personality
    jane_personality = debater.get_ai_personality("Jane")
    assert "warm" in jane_personality.lower()
    assert "empathetic" in jane_personality.lower()
    assert "heart" in jane_personality.lower()
    assert "conversational" in jane_personality.lower()
    
    # Test Valentino's personality
    valentino_personality = debater.get_ai_personality("Valentino")
    assert "brilliant" in valentino_personality.lower()
    assert "intellectual" in valentino_personality.lower()
    assert "superior" in valentino_personality.lower()
    assert "concise" in valentino_personality.lower()

def test_debate_history_queue_limit():
    debater = AIDebater()
    
    # Test that history gets truncated after max size
    ground_statement = "Test statement"
    debater.ground_statement = ground_statement
    
    # Add more than MAX_HISTORY entries to debate history
    for i in range(25):  # More than MAX_HISTORY
        debater.debate_history.append(f"Response {i}")
    
    # Check if size is limited to MAX_HISTORY
    assert len(debater.debate_history) == debater.MAX_HISTORY
    
    # Check if ground statement is preserved
    assert debater.ground_statement == ground_statement
    
    # Check the actual debate history content
    history_list = list(debater.debate_history)
    assert len(history_list) == debater.MAX_HISTORY
    assert history_list[-1] == "Response 24"  # Latest response should be preserved

@patch('openai.OpenAI')
def test_model_settings(mock_openai, debater):
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    )
    mock_openai.return_value = mock_client
    
    debater.client = mock_client
    debater.generate_response("test prompt", "Jane")
    
    # Check if correct model and token limit were used
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-4o-mini"
    assert call_args["max_tokens"] == 500

# New test for the video generation functionality
@patch('ai_debate.asyncio.run')
@patch('ai_debate.process_debate')
@patch('ai_debate.create_debate_video')
@patch('openai.OpenAI')
def test_video_generation_after_audio(mock_openai, mock_create_video, mock_process_debate, 
                                     mock_asyncio_run, debater, mock_openai_response):
    mock_client = Mock()
    responses = [
        mock_openai_response,  # Jane's first response
        mock_openai_response,  # Valentino's first response
        Mock(choices=[Mock(message=Mock(content="surrender"))])  # Jane surrenders
    ]
    mock_client.chat.completions.create.side_effect = responses
    mock_openai.return_value = mock_client
    
    debater.client = mock_client
    ground_statement = "Test statement"
    
    # Mock file operations
    with patch('builtins.open', MagicMock()):
        # Run debate with audio generation
        results = debater.debate(ground_statement, generate_audio=True)
    
    # Verify that both audio and video generation functions were called
    mock_asyncio_run.assert_called_once()
    mock_create_video.assert_called_once()

@patch('ai_debate.asyncio.run')
@patch('ai_debate.process_debate')
@patch('ai_debate.create_debate_video')
@patch('os.path.exists')
def test_existing_debate_file_for_video(mock_exists, mock_create_video, mock_process_debate, mock_asyncio_run, debater):
    # Make os.path.exists return True for debate.txt
    mock_exists.return_value = True
    
    # Mock file operations
    mock_file_content = MagicMock()
    mock_file_content.__iter__.return_value = [
        "Ground Statement: Test existing statement\n",
        "AI Debater 1: Test response 1\n",
        "AI Debater 2: Test response 2\n"
    ]
    
    with patch('builtins.open', MagicMock(return_value=mock_file_content)):
        # Use the existing file to generate audio and video
        debater.debate("", use_existing=True, generate_audio=True)
    
    # Verify both audio and video generation were called
    mock_asyncio_run.assert_called_once()
    mock_create_video.assert_called_once()
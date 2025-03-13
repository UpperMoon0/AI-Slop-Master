import pytest
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from collections import deque
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
def test_debate_flow(mock_openai, debater, mock_openai_response):
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    mock_openai.return_value = mock_client
    
    debater.client = mock_client
    ground_statement = "Test statement"
    results = debater.debate(ground_statement, max_rounds=1, generate_audio=False)
    
    assert results[0] == ground_statement
    assert len(results) == 3  # Ground statement + 2 AI responses
    assert mock_client.chat.completions.create.call_count == 2

@patch('openai.OpenAI')
def test_surrender_condition(mock_openai, debater):
    mock_client = Mock()
    surrender_response = Mock(choices=[Mock(message=Mock(content="I surrender"))])
    mock_client.chat.completions.create.return_value = surrender_response
    mock_openai.return_value = mock_client
    
    debater.client = mock_client
    results = debater.debate("Test statement", max_rounds=3, generate_audio=False)
    
    assert len(results) == 2  # Ground statement + surrender response
    assert "surrender" in results[-1].lower()

@patch('openai.OpenAI')
def test_debate_file_output(mock_openai, debater, mock_openai_response):
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    mock_openai.return_value = mock_client
    
    debater.client = mock_client
    ground_statement = "Test statement"
    
    # Delete debate.txt if it exists
    if os.path.exists('debate.txt'):
        os.remove('debate.txt')
    
    results = debater.debate(ground_statement, max_rounds=1, generate_audio=False)
    
    # Check if file exists and contains correct content
    assert os.path.exists('debate.txt')
    with open('debate.txt', 'r', encoding='utf-8') as f:
        content = f.read().splitlines()
    
    assert content[0] == f"Ground Statement: {ground_statement}"
    assert len(content) == 3  # Ground statement + 2 AI responses
    assert content[1].startswith("AI Debater 1: ")
    assert content[2].startswith("AI Debater 2: ")
    
    # Clean up
    os.remove('debate.txt')

@patch('openai.OpenAI')
def test_debate_history_in_context(mock_openai, debater):
    mock_client = Mock()
    responses = [
        Mock(choices=[Mock(message=Mock(content="First response"))]),
        Mock(choices=[Mock(message=Mock(content="Second response"))])
    ]
    mock_client.chat.completions.create.side_effect = responses
    mock_openai.return_value = mock_client
    
    debater.client = mock_client
    ground_statement = "Test statement"
    debater.debate(ground_statement, max_rounds=1, generate_audio=False)
    
    # Get the calls made to the API
    calls = mock_client.chat.completions.create.call_args_list
    
    # Check second API call includes previous context
    second_call = calls[1][1]['messages'][0]['content']
    assert "Ground Statement: Test statement" in second_call
    assert "AI Debater 1: First response" in second_call

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
    system_message = mock_client.chat.completions.create.call_args[1]['messages'][0]['content']  # Fixed: accessing dict key directly
    assert "surrender" in system_message.lower()
    assert "If you find your position impossible to defend" in system_message

def test_ai_personality_differences():
    debater = AIDebater()
    
    # Test AI 1's personality
    ai1_personality = debater.get_ai_personality("AI Debater 1")
    assert "warm" in ai1_personality.lower()
    assert "emotional" in ai1_personality.lower()
    assert "heart" in ai1_personality.lower()
    assert "conversational" in ai1_personality.lower()
    
    # Test AI 2's personality
    ai2_personality = debater.get_ai_personality("AI Debater 2")
    assert "brilliant" in ai2_personality.lower()
    assert "intellectual" in ai2_personality.lower()
    assert "sophisticated" in ai2_personality.lower()
    assert "scholar" in ai2_personality.lower()

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
    debater.generate_response("test prompt", "AI Debater 1")
    
    # Check if correct model and token limit were used
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-4o-mini"
    assert call_args["max_tokens"] == 200
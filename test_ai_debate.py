import pytest
from unittest.mock import Mock, patch, mock_open
from ai_debate import AIDebater

@pytest.fixture
def ai_debater():
    """Create a fresh AIDebater instance for each test."""
    with patch('ai_debate.OpenAI') as mock_openai:
        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        debater = AIDebater()
        yield debater

def test_ai_debater_initialization(ai_debater):
    """Test AIDebater initialization."""
    assert ai_debater.debate_history is not None
    assert ai_debater.ground_statement is None
    assert ai_debater.current_speaker_number == 1

def test_get_ai_personality_jane():
    """Test getting Jane's personality."""
    debater = AIDebater()
    jane_personality = debater.get_ai_personality("Jane")
    assert "Jane" in jane_personality
    assert "warm" in jane_personality.lower()
    assert "empathetic" in jane_personality.lower()

def test_get_ai_personality_valentino():
    """Test getting Valentino's personality."""
    debater = AIDebater()
    valentino_personality = debater.get_ai_personality("Valentino")
    assert "Valentino" in valentino_personality
    assert "intellectual" in valentino_personality.lower()
    assert "concise" in valentino_personality.lower()

def test_check_similarity():
    """Test text similarity checking."""
    debater = AIDebater()
    text1 = "AI helps create art"
    text2 = "AI assists in creating art"  # More similar text pair
    text3 = "The weather is nice today"
    
    assert debater._check_similarity(text1, text2)  # Should be similar
    assert not debater._check_similarity(text1, text3)  # Should not be similar

@pytest.mark.parametrize("text1,text2,expected", [
    ("hello world", "hello earth", True),  # Similar texts
    ("hello world", "goodbye moon", False),  # Different texts
    ("", "", False),  # Empty texts
    ("hello", "", False),  # One empty text
])
def test_check_similarity_parametrized(ai_debater, text1, text2, expected):
    """Test text similarity with different inputs."""
    assert ai_debater._check_similarity(text1, text2) == expected

@patch('ai_debate.OpenAI')
def test_generate_response(mock_openai):
    """Test response generation."""
    # Setup mock
    mock_client = Mock()
    mock_openai.return_value = mock_client
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_client.chat.completions.create.return_value = mock_response
    
    debater = AIDebater()
    response = debater.generate_response("Test prompt", "Jane")
    
    assert response == "Test response"
    mock_client.chat.completions.create.assert_called_once()

def test_surrender_detection():
    """Test surrender phrase detection."""
    # Set up OpenAI mock before creating AIDebater instance
    with patch('ai_debate.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="i surrender"))]
        )
        mock_openai.return_value = mock_client
        
        # Create AIDebater instance after mock is set up
        debater = AIDebater()
        
        # Generate response and verify
        response = debater.generate_response("Test prompt", "Jane")
        assert response == "surrender"
        mock_client.chat.completions.create.assert_called_once()

def test_debate_with_existing_file(tmp_path):
    """Test debate using existing debate.txt file."""
    # Create a temporary debate.txt
    debate_file = tmp_path / "debate.txt"
    content = """Narrator: Welcome to our AI debate.
Ground Statement: Test statement
AI Debater 1: Test argument
AI Debater 2: Counter argument"""
    debate_file.write_text(content)
    
    with patch('ai_debate.open', mock_open(read_data=content)):
        with patch('asyncio.run') as mock_run:  # Mock asyncio.run instead of process_debate
            with patch('ai_debate.create_debate_video'):
                debater = AIDebater()
                results = debater.debate("Test", use_existing=True)
                assert len(results) > 0
                mock_run.assert_called_once()

def test_generate_debate():
    """Test debate text generation."""
    debater = AIDebater()
    debater.ground_statement = "Test statement"
    
    with patch('builtins.open', create=True) as mock_open:
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        debate_text = debater.generate_debate()
        
        assert "Welcome to our AI debate" in debate_text
        assert "Test statement" in debate_text
        mock_file.write.assert_called()
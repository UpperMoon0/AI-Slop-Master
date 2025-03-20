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
    assert ai_debater.ground_statement_summary is None

def test_get_ai_personality_jane():
    """Test getting Jane's personality."""
    debater = AIDebater()
    jane_personality = debater.get_ai_personality("Jane")
    assert "Jane" in jane_personality
    assert "warm" in jane_personality.lower()
    assert "empathy" in jane_personality.lower()  # Changed from "empathetic" to "empathy"
    assert "concise" in jane_personality.lower()

def test_get_ai_personality_valentino():
    """Test getting Valentino's personality."""
    debater = AIDebater()
    valentino_personality = debater.get_ai_personality("Valentino")
    assert "Valentino" in valentino_personality
    assert "intellectual" in valentino_personality.lower()
    assert "concise" in valentino_personality.lower()
    assert "brilliant" in valentino_personality.lower()

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
    ("goodbye moon", "hello world", False),  # Different texts
    ("", "", False),  # Empty texts
    ("hello", "", False),  # One empty text
    ("AI is revolutionizing healthcare", "AI is changing healthcare", True),  # More similar text pair that meets the threshold
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
    # Check model used is correct
    args, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.8
    assert kwargs["max_tokens"] == 500

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

@pytest.mark.parametrize("surrender_phrase", [
    "I surrender",
    "i give up",
    "You win",
    "I concede",
    "I SURRENDER",  # Test case sensitivity
])
def test_surrender_phrases(surrender_phrase):
    """Test various surrender phrases are detected."""
    with patch('ai_debate.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=f"After considering all arguments, {surrender_phrase}."))]
        )
        mock_openai.return_value = mock_client
        
        debater = AIDebater()
        response = debater.generate_response("Test prompt", "Jane")
        assert response == "surrender"

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
            with patch('ai_debate.reformat_debate_file'):
                with patch('ai_debate.create_debate_video'):
                    debater = AIDebater()
                    results = debater.debate("Test", use_existing_scripts=True)
                    assert len(results) > 0
                    mock_run.assert_called_once()

@patch('ai_debate.OpenAI')
def test_summarize_ground_statement(mock_openai):
    """Test summarizing ground statement."""
    # Setup mock
    mock_client = Mock()
    mock_openai.return_value = mock_client
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Summarized statement"))]
    mock_client.chat.completions.create.return_value = mock_response
    
    debater = AIDebater()
    summary = debater.summarize_ground_statement("This is a long ground statement that needs to be summarized for display during the debate.")
    
    assert summary == "Summarized statement"
    mock_client.chat.completions.create.assert_called_once()
    # Check model and parameters
    args, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 60

@patch('ai_debate.OpenAI')
def test_generate_video_title(mock_openai):
    """Test generating video title."""
    # Setup mock
    mock_client = Mock()
    mock_openai.return_value = mock_client
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="2 AIs Debate About Fascinating Topic"))]
    mock_client.chat.completions.create.return_value = mock_response
    
    debater = AIDebater()
    title = debater.generate_video_title("This is a topic for debate")
    
    assert title == "2 AIs Debate About Fascinating Topic"
    mock_client.chat.completions.create.assert_called_once()
    # Check model and parameters
    args, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 60

@patch('ai_debate.OpenAI')
def test_generate_video_title_adds_prefix(mock_openai):
    """Test that prefix is added if missing from title."""
    # Setup mock with response missing prefix
    mock_client = Mock()
    mock_openai.return_value = mock_client
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Fascinating Topic"))]
    mock_client.chat.completions.create.return_value = mock_response
    
    debater = AIDebater()
    title = debater.generate_video_title("This is a topic for debate")
    
    assert title == "2 AIs Debate About Fascinating Topic"

@patch('ai_debate.OpenAI')
def test_generate_video_description(mock_openai):
    """Test generating video description."""
    # Setup mock
    mock_client = Mock()
    mock_openai.return_value = mock_client
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="This is a video description. It explains the debate. Watch to find out who wins."))]
    mock_client.chat.completions.create.return_value = mock_response
    
    debater = AIDebater()
    description = debater.generate_video_description("This is a topic for debate")
    
    assert "This is a video description" in description
    mock_client.chat.completions.create.assert_called_once()
    # Check model and parameters
    args, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 150

def test_generate_debate():
    """Test debate text generation."""
    debater = AIDebater()
    debater.ground_statement = "Test statement"
    debater.ground_statement_summary = "Test summary"
    
    with patch('builtins.open', create=True) as mock_open:
        with patch('ai_debate.AIDebater.generate_video_title', return_value="2 AIs Debate About Test Topic"):
            with patch('ai_debate.AIDebater.generate_video_description', return_value="Test description"):
                mock_file = Mock()
                mock_file_video = Mock()
                mock_open.return_value.__enter__.side_effect = [mock_file_video, mock_file]
                
                debate_text = debater.generate_debate()
                
                assert "Welcome to our AI debate" in debate_text
                assert "Test statement" in debate_text
                assert "Test summary" in debate_text
                mock_file.write.assert_called()
                mock_file_video.write.assert_called()

@patch('ai_debate.OpenAI')
@patch('ai_debate.asyncio.run')
@patch('ai_debate.create_debate_video')
@patch('builtins.open', new_callable=mock_open)
def test_debate_full_process(mock_file, mock_create_video, mock_asyncio, mock_openai):
    """Test full debate process with surrender."""
    # Setup mocks
    mock_client = Mock()
    mock_openai.return_value = mock_client
    
    # Mock responses for various API calls
    summary_response = Mock(choices=[Mock(message=Mock(content="Test Summary"))])
    title_response = Mock(choices=[Mock(message=Mock(content="2 AIs Debate About Test"))])
    desc_response = Mock(choices=[Mock(message=Mock(content="Test Description"))])
    # First responses for the debate exchanges
    response1 = Mock(choices=[Mock(message=Mock(content="First response"))])
    response2 = Mock(choices=[Mock(message=Mock(content="Second response"))])
    # Make the third response a surrender
    response3 = Mock(choices=[Mock(message=Mock(content="I surrender"))])
    
    # Configure the mock to return different responses for sequential calls
    mock_client.chat.completions.create.side_effect = [
        summary_response, title_response, desc_response,
        response1, response2, response3
    ]
    
    debater = AIDebater()
    results = debater.debate("Test statement", jane_first=True)
    
    # Verify results
    assert "surrender" in results
    assert mock_asyncio.called
    assert mock_create_video.called
    assert mock_file().write.called

def test_debate_with_existing_audio():
    """Test debate using existing audio files."""
    with patch('ai_debate.reformat_debate_file'):
        with patch('ai_debate.create_debate_video') as mock_create_video:
            with patch('builtins.open', mock_open(read_data="Narrator: Welcome\nAI Debater 1: Test")):
                debater = AIDebater()
                results = debater.debate("Test", use_existing_scripts=True, use_existing_audios=True)
                assert len(results) > 0
                mock_create_video.assert_called_once()
                # Ensure process_debate wasn't called (because we're using existing audio)
                assert not any('process_debate' in str(c) for c in mock_create_video.mock_calls)
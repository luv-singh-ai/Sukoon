import pytest
import logging
from datetime import datetime
import os
from unittest.mock import Mock, patch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sukoon import (
    run_conversational_agent,
    run_suicide_prevention_agent,
    run_anger_management_agent,
    run_motivational_agent,
    run_dialectical_behavior_therapy_agent,
    run_cognitive_behavioral_therapy_agent,
    run_denial_agent,
    State,
)
from langchain_core.messages import HumanMessage

# Set up logging
def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/test_agents_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

@pytest.fixture
def mock_state():
    return State(messages=[HumanMessage(content="Test message")])

@pytest.fixture
def mock_model():
    with patch('sukoon.model_a') as mock:
        mock_instance = Mock()
        mock_instance.invoke.return_value = "Test response"
        mock.return_value = mock_instance
        yield mock

def test_conversational_agent(mock_state, caplog):
    logger.info("Testing conversational agent")
    
    result = run_conversational_agent(mock_state)
    
    # Log the response
    logger.info(f"Conversational agent response: {result}")
    
    # Basic checks
    assert isinstance(result, dict)
    assert "messages" in result
    assert result["messages"] is not None
    assert isinstance(str(result["messages"]), str)

def test_suicide_prevention_agent(mock_state, caplog):
    logger.info("Testing suicide prevention agent")
    
    result = run_suicide_prevention_agent(mock_state)
    
    logger.info(f"Suicide prevention agent response: {result}")
    
    assert isinstance(result, dict)
    assert "messages" in result
    assert result["messages"] is not None
    assert isinstance(str(result["messages"]), str)

def test_anger_management_agent(mock_state, caplog):
    logger.info("Testing anger management agent")
    
    result = run_anger_management_agent(mock_state)
    
    logger.info(f"Anger management agent response: {result}")
    
    assert isinstance(result, dict)
    assert "messages" in result
    assert result["messages"] is not None
    assert isinstance(str(result["messages"]), str)

def test_motivational_agent(mock_state, caplog):
    logger.info("Testing motivational agent")
    
    result = run_motivational_agent(mock_state)
    
    logger.info(f"Motivational agent response: {result}")
    
    assert isinstance(result, dict)
    assert "messages" in result
    assert result["messages"] is not None
    assert isinstance(str(result["messages"]), str)

def test_dbt_agent(mock_state, caplog):
    logger.info("Testing DBT agent")
    
    result = run_dialectical_behavior_therapy_agent(mock_state)
    
    logger.info(f"DBT agent response: {result}")
    
    assert isinstance(result, dict)
    assert "messages" in result
    assert result["messages"] is not None
    assert isinstance(str(result["messages"]), str)

def test_cbt_agent(mock_state, caplog):
    logger.info("Testing CBT agent")
    
    result = run_cognitive_behavioral_therapy_agent(mock_state)
    
    logger.info(f"CBT agent response: {result}")
    
    assert isinstance(result, dict)
    assert "messages" in result
    assert result["messages"] is not None
    assert isinstance(str(result["messages"]), str)

def test_denial_agent(mock_state, caplog):
    logger.info("Testing denial agent")
    
    result = run_denial_agent(mock_state)
    
    logger.info(f"Denial agent response: {result}")
    
    assert isinstance(result, dict)
    assert "messages" in result
    assert isinstance(result["messages"], str)
    assert "Please use this wisely" in result["messages"]

if __name__ == "__main__":
    pytest.main(["-v", __file__])

# NOTE: PREVIOUS VERSION
# import pytest
# import logging
# import os
# from datetime import datetime
# from unittest.mock import Mock, patch
# from sukoon import route_query, run_conversational_agent, run_suicide_prevention_agent, chat, State
# from langchain_core.messages import HumanMessage, AIMessage
# from _pytest.logging import caplog
# import json
# from typing import Any

# # Custom JSON encoder for handling AIMessage and HumanMessage objects
# class MessageEncoder(json.JSONEncoder):
#     def default(self, obj: Any) -> Any:
#         if isinstance(obj, (AIMessage, HumanMessage)):
#             return {
#                 'type': obj.__class__.__name__,
#                 'content': obj.content,
#                 'additional_kwargs': obj.additional_kwargs,
#             }
#         return super().default(obj)

# def serialize_message(obj: Any) -> dict:
#     """Helper function to serialize message objects."""
#     if isinstance(obj, (AIMessage, HumanMessage)):
#         return {
#             'type': obj.__class__.__name__,
#             'content': obj.content,
#             'additional_kwargs': obj.additional_kwargs,
#         }
#     elif isinstance(obj, dict):
#         return {k: serialize_message(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [serialize_message(item) for item in obj]
#     return obj

# # Set up logging configuration
# def setup_logging():
#     if not os.path.exists('logs'):
#         os.makedirs('logs')
    
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     log_file = f'logs/test_run_{timestamp}.log'
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)

# logger = setup_logging()

# @pytest.fixture(autouse=True)
# def log_test_info(caplog):
#     caplog.set_level(logging.INFO)
#     yield

# @pytest.fixture
# def mock_openai():
#     with patch('sukoon.ChatOpenAI') as mock:
#         mock_instance = Mock()
#         # Mock the invoke method to return a simple string
#         mock_instance.invoke.return_value = "Test response"
#         mock.return_value = mock_instance
#         yield mock

# @pytest.fixture
# def mock_state():
#     return State(messages=[HumanMessage(content="Test message")])

# def log_json(data: dict) -> None:
#     """Helper function to safely log JSON data."""
#     try:
#         serialized_data = serialize_message(data)
#         logger.info(json.dumps(serialized_data, indent=2))
#     except Exception as e:
#         logger.error(f"Error serializing data: {e}")
#         logger.info(f"Raw data: {data}")

# def test_route_query(mock_openai, mock_state, caplog):
#     logger.info("Starting route_query test")
    
#     mock_structured = Mock()
#     mock_openai.return_value.with_structured_output.return_value = mock_structured
    
#     # Test case 1: Happy path
#     mock_structured.invoke.return_value.route = "conversational"
#     result = route_query(mock_state)
#     log_json({
#         "test_case": "happy_path",
#         "input": serialize_message(mock_state['messages'][-1]),
#         "route": result
#     })
#     assert result == "conversational"
    
#     # Test case 2: Concerning content
#     mock_state_concern = State(messages=[HumanMessage(content="I'm thinking of ending it all")])
#     mock_structured.invoke.return_value.route = "suicide_prevention"
#     result = route_query(mock_state_concern)
#     log_json({
#         "test_case": "concerning_content",
#         "input": serialize_message(mock_state_concern['messages'][-1]),
#         "route": result
#     })
#     assert result == "suicide_prevention"

# def test_run_conversational_agent(mock_openai, mock_state, caplog):
#     logger.info("Starting conversational agent test")
    
#     # Mocked response
#     expected_response = "Hello! How can I assist you today?"
#     mock_openai.return_value.invoke.return_value = expected_response
    
#     result = run_conversational_agent(mock_state)
#     log_json({
#         "test": "conversational_agent",
#         "input": serialize_message(mock_state['messages'][-1]),
#         "output": result
#     })
#     assert isinstance(result, dict)
#     assert "messages" in result
#     assert isinstance(result["messages"], list)
#     assert len(result["messages"]) > 0
#     assert isinstance(result["messages"][-1], AIMessage)
#     assert isinstance(result["messages"][-1].content, str)
#     assert len(result["messages"][-1].content) > 0

# def test_run_suicide_prevention_agent(mock_openai, mock_state, caplog):
#     logger.info("Starting suicide prevention agent test")
    
#     # Mocked response
#     expected_response = "I'm sorry to hear that you're feeling this way. Please tell me more about what's going on."
#     mock_openai.return_value.invoke.return_value = expected_response
    
#     result = run_suicide_prevention_agent(mock_state)
#     log_json({
#         "test": "suicide_prevention_agent",
#         "input": serialize_message(mock_state['messages'][-1]),
#         "output": result
#     })
#     assert isinstance(result, dict)
#     assert "messages" in result
#     assert isinstance(result["messages"], list)
#     assert len(result["messages"]) > 0
#     assert isinstance(result["messages"][-1], AIMessage)
#     assert isinstance(result["messages"][-1].content, str)
#     assert len(result["messages"][-1].content) > 0

# @pytest.mark.parametrize("input_message,expected_route", [
#     (
#         "I'm feeling happy today",
#         "conversational"
#     ),
#     (
#         "I'm thinking about ending it all",
#         "suicide_prevention"
#     ),
# ])
# def test_chat_routing(mock_openai, input_message, expected_route, caplog):
#     logger.info(f"Starting chat routing test with input: {input_message}")
    
#     mock_structured = Mock()
#     mock_openai.return_value.with_structured_output.return_value = mock_structured
#     mock_structured.invoke.return_value.route = expected_route
    
#     # Mock response from the chatbot
#     mock_response = "Some response"
#     mock_openai.return_value.invoke.return_value = mock_response
    
#     config = {"configurable": {"thread_id": "test"}}
#     response = chat(input_message, config)
    
#     log_json({
#         "test": "chat_routing",
#         "input": input_message,
#         "route": expected_route,
#         "output": response
#     })
    
#     # Verify that we get a non-empty string response
#     assert isinstance(response, AIMessage)
#     assert isinstance(response.content, str)
#     # assert len(response) > 0

# def test_error_handling(mock_openai, mock_state, caplog):
#     logger.info("Starting error handling test")
    
#     error_message = "API Error"
#     mock_openai.return_value.invoke.side_effect = Exception(error_message)
    
#     with pytest.raises(Exception) as exc_info:
#         run_conversational_agent(mock_state)
#     log_json({
#         "test": "error_handling",
#         "input": serialize_message(mock_state['messages'][-1]),
#         "error": str(exc_info.value)
#     })
#     assert str(exc_info.value) == error_message

# if __name__ == "__main__":
#     pytest.main([
#         __file__,
#         "-v",
#         "--html=test_reports/report.html",
#         "--capture=tee-sys"
#     ])
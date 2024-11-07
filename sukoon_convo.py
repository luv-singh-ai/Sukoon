from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from typing import TypedDict, Literal, Annotated, List, Dict
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel, Field

import os
import yaml
import json
from datetime import datetime
import pandas as pd
import sqlite3
from typing import List, Dict
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from groq import Groq
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_CHAT_MODEL") # llama3-70b , llama3.1-450B

# Define the state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

openai_api_key = os.getenv("OPENAI_API_KEY")

# Load prompts from YAML
def load_prompts(file_path='prompts.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

prompts = load_prompts()

# Initialize OpenAI model
model = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Define prompts for different agents
# planner_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a planner agent that decides which specialized agent to call based on the user's input. Respond with one of 'suicide_prevention', 'conversational', 'anger_management', 'motivational', or 'mindfulness' based on the user's emotion."),
#     ("human", "{input}"),
# ])

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", prompts['planner_agent_prompt']),
    ("human", "{input}"),
])

conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", prompts['empathetic_agent_prompt']),
    ("human", "{input}"),
])

suicide_prevention_prompt = ChatPromptTemplate.from_messages([
    ("system", prompts['suicide_prevention_agent_prompt']),
    ("human", "{input}"),
])

anger_management_prompt = ChatPromptTemplate.from_messages([
    ("system", prompts['anger_prevention_agent_prompt']),
    ("human", "{input}"),
])

motivational_prompt = ChatPromptTemplate.from_messages([
    ("system", prompts['motivational_agent_prompt']),
    ("human", "{input}"),
])

# mindfulness_prompt = ChatPromptTemplate.from_messages([
#     ("system", prompts['mindfulness_agent_prompt']),
#     ("human", "{input}"),
# ])

dialectical_behavior_therapy_prompt = ChatPromptTemplate.from_messages([
    ("system", prompts['dbt_agent_prompt']),
    ("human", "{input}")
])

cognitive_behavioral_therapy_prompt = ChatPromptTemplate.from_messages([
    ("system", prompts['cbt_agent_prompt']),
    ("human", "{input}")
])

# Define router
def route_query(state: State):
  
    class RouteQuery(BaseModel):
        """Route a user query to the most relevant node based on the emotional or psychological state identified from the query intent."""
        
        route: Literal[
            "conversational", "suicide_prevention", "anger_management", 
            "motivational", "dialectical_behavior_therapy", "cognitive_behavioral_therapy"
        ] = Field(
            ...,
            description=(
                "Choose the most appropriate agent based on the user's emotional or psychological needs, inferred from their dialogue: "

                "'conversational' is ideal for users seeking general empathetic interaction, companionship, or simply wishing to engage in casual dialogue. This route aims to provide emotional support through open, non-directive conversation. \n"
                "Example: A user says, 'I've been feeling a bit lonely lately. I just need someone to talk to about my day.'\n"

                "'suicide_prevention' is critical for users who express thoughts of hopelessness, self-harm, suicidal ideation, or severe emotional distress. This route provides immediate intervention, offering resources and support to de-escalate the crisis. \n"
                "Example: A user states, 'I feel like no one would care if I were gone. I don't want to keep going anymore.'\n"

                "'anger_management' should be selected for users expressing frustration, irritability, or anger. This route helps the user manage their temper, process their emotions constructively, and reduce the risk of conflict escalation. \n"
                "Example: A user vents, 'I'm so mad at my boss! He keeps undermining me, and I'm about to explode.'\n"

                "'motivational' is suited for users who feel demotivated, struggle with low self-esteem, or are seeking encouragement to pursue their goals. This route offers positive reinforcement and practical strategies for improving self-worth and maintaining focus. \n"
                "Example: A user shares, 'I’ve been feeling stuck. Every time I try to work on my project, I lose motivation. What’s the point of even trying?' \n"

                "'dialectical_behavior_therapy' (DBT) should be used for users dealing with intense, fluctuating emotions or feeling emotionally overwhelmed. DBT teaches skills for emotional regulation, distress tolerance, and managing interpersonal relationships. \n"
                "Example: A user says, 'One moment I’m okay, but then I’m hit with this overwhelming sadness and anger. I don’t know how to control my emotions.'\n"

                "'cognitive_behavioral_therapy' (CBT) is appropriate for users struggling with negative or distorted thinking patterns, self-criticism, or irrational beliefs. CBT helps them reframe unhealthy thoughts into more positive, balanced perspectives. \n"
                "Example: A user confides, 'I always mess things up. No matter what I do, I feel like a failure, and it’s hard to think any differently.'"
            )
        )

    structured_llm_router = model.with_structured_output(RouteQuery)
    question_router = planner_prompt | structured_llm_router
    last_message = state["messages"][-1]
    resp = question_router.invoke({"input": last_message})
    return resp.route

# Define agent functions
def run_conversational_agent(state: State):
    print("Running conversational agent")
    convo_model = conversational_prompt | model
    response = convo_model.invoke(state["messages"])
    return {"messages": response}

def run_suicide_prevention_agent(state: State):
    print("Running suicide prevention agent")
    concern_model = suicide_prevention_prompt | model
    response = concern_model.invoke(state["messages"])
    return {"messages": response}

def run_anger_management_agent(state: State):
    print("Running anger management agent")
    anger_model = anger_management_prompt | model
    response = anger_model.invoke(state["messages"])
    return {"messages": response}

def run_motivational_agent(state: State):
    print("Running motivational agent")
    motivation_model = motivational_prompt | model
    response = motivation_model.invoke(state["messages"])
    return {"messages": response}

# def run_mindfulness_agent(state: State):
#     print("Running mindfulness agent")
#     mindfulness_model = mindfulness_prompt | model
#     response = mindfulness_model.invoke(state["messages"])
#     return {"messages": response}

def run_dialectical_behavior_therapy_agent(state: State):
    print("Running dialectical_behavior_therapy agent")
    dialectical_behavior_therapy_model = dialectical_behavior_therapy_prompt | model
    response = dialectical_behavior_therapy_model.invoke(state["messages"])
    return {"messages": response}

def run_cognitive_behavioral_therapy_agent(state: State):
    print("Running cognitive_behavioral_therapy agent")
    cognitive_behavioral_therapy_model = cognitive_behavioral_therapy_prompt | model
    response = cognitive_behavioral_therapy_model.invoke(state["messages"])
    return {"messages": response}

# Create the graph
workflow = StateGraph(State)

# Add nodes for each agent
workflow.add_node("conversational", run_conversational_agent)
workflow.add_node("suicide_prevention", run_suicide_prevention_agent)
workflow.add_node("anger_management", run_anger_management_agent)
workflow.add_node("motivational", run_motivational_agent)
# workflow.add_node("mindfulness", run_mindfulness_agent)
workflow.add_node("dialectical_behavior_therapy", run_dialectical_behavior_therapy_agent)
workflow.add_node("cognitive_behavioral_therapy", run_cognitive_behavioral_therapy_agent)

# Add edges
workflow.add_conditional_edges(
    START,
    route_query,
    {
        "conversational": "conversational",
        "suicide_prevention": "suicide_prevention",
        "anger_management": "anger_management",
        "motivational": "motivational",
        # "mindfulness": "mindfulness",
        "dialectical_behavior_therapy": "dialectical_behavior_therapy",
        "cognitive_behavioral_therapy": "cognitive_behavioral_therapy"
    }
)

workflow.add_edge("conversational", END)
workflow.add_edge("suicide_prevention", END)
workflow.add_edge("anger_management", END)
workflow.add_edge("motivational", END)
# workflow.add_edge("mindfulness", END)
workflow.add_edge("dialectical_behavior_therapy", END)
workflow.add_edge("cognitive_behavioral_therapy", END)

# Compile the graph
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


class ConversationManager:
    def __init__(self, groq_api_key: str, groq_model: str):
        # a new chat completion client to act as student persona using Groq API
        self.client = Groq(api_key = groq_api_key)
        self.groq_model = groq_model
        self.conversation_history = []
        self.current_context = []
        # Create personas directory if it doesn't exist
        self.personas_dir = Path("personas")
        # self.personas_dir.mkdir(exist_ok=True)
        # create personas dictionary
        self.personas = self._load_personas()
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with updated schema"""
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
        # Check if column exists
            cursor.execute("PRAGMA table_info(conversations)")
            columns = [col[1] for col in cursor.fetchall()]
        
        # Add missing columns if needed
        if 'agent_type' not in columns:
            cursor.execute('ALTER TABLE conversations ADD COLUMN agent_type TEXT')
        if 'conversation_id' not in columns:
            cursor.execute('ALTER TABLE conversations ADD COLUMN conversation_id TEXT')
            
        else:
            # Create new table with all columns
            cursor.execute('''
                CREATE TABLE conversations (
                    turn INTEGER,
                    timestamp TEXT,
                    student TEXT,
                    sukoon TEXT,
                    agent_type TEXT,
                    conversation_id TEXT
                )
            ''')
        conn.commit()
        conn.close()
    
    def _get_conversation_history(self, limit: int = 3) -> str:
        """Retrieve recent conversation history formatted for context"""
        history = []
        for turn in self.current_context[-limit:]:
            history.append(f"Student: {turn['student']}")
            history.append(f"Sukoon: {turn['sukoon']}")
        return "\n".join(history)
    
    def _load_personas(self) -> Dict[str, str]:
        # load all personas files from persona dcitionary
        personas = {}
        for persona_file in self.personas_dir.glob("*.txt"):
            personas[persona_file.stem] = persona_file.read_text()
        return personas
    
    def generate_persona_message(self, persona_name: str) -> str:
        try:
            # Get base persona prompt
            base_prompt = self.personas.get(persona_name, self.personas["rohan"])
            
            # Add conversation history to context
            context = self._get_conversation_history()
            enhanced_prompt = f"{base_prompt}\n#Previous conversation:\n{context}\n\nNow start talking, maintaning context:"
            
            system_prompt = {
                "role" : "system",
                # "content": self.personas.get(persona_name, self.personas["rohan"])
                "content": enhanced_prompt
            }
            chat_completion = self.client.chat.completions.create(
                messages = [system_prompt],
                model=self.groq_model,
                temperature=0.9,
                max_tokens=150
            )
            response = chat_completion.choices[0].message.content
            return str(response)
        except Exception as e:
            print(f"Error generating persona message: {e}")
            return "Please continue talking"
        
    def chat(self, message: str, config: dict) -> tuple[str, str]:
            try:
                # Add conversation history to the message context
                context = self._get_conversation_history()
                contextual_message = f"{message}\nPrevious conversation has been added here as reference:\n{context}"
                
                result = graph.invoke(
                    {"messages": [HumanMessage(content=contextual_message)]}, 
                    config=config
                )
                
                # Extract agent type from memory checkpoint
                agent_type = self._extract_agent_type(memory)
                
                return result["messages"][-1].content, agent_type
            except Exception as e:
                print(f"Error in chat: {e}")
                return "Please continue talking", "unknown"
    
    def _extract_agent_type(self, memory) -> str:
        """Extract the agent type from the memory checkpoint"""
        try:
            # Access the memory checkpoint to find which agent was called
            # This might need adjustment based on your specific memory structure
            checkpoint = memory.get_checkpoint()
            for key in checkpoint.keys():
                if any(agent in key for agent in [
                    "conversational", "suicide_prevention", "anger_management",
                    "motivational", "dialectical_behavior_therapy", 
                    "cognitive_behavioral_therapy"
                ]):
                    return key.split("/")[0]  # Extract agent name from checkpoint key
            return "unknown"
        except:
            return "unknown"
    
    def save_conversation(self, format: str = "sqlite"):
        if format == "sqlite":
            conn = sqlite3.connect('conversations.db')
            df = pd.DataFrame(self.conversation_history)
            df.to_sql("conversations", conn, if_exists='append', index=False)
            conn.close()
        
        elif format == "json":
            # Prepare conversation data
            conversation_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": self.conversation_history
            }
            
            with open(f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(conversation_data, f, indent = 2)
        else:
            df = pd.DataFrame(self.conversation_history)
            df.to_csv(f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index = False)

    def run_conversation(self, max_turns: int = 5):
        config = {"configurable": {"thread_id": "1"}}
        conversation_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for i in range(max_turns):
            # Generate student message
            user_input = self.generate_persona_message("rohan")
            print(f"\nStudent: {user_input}")
            
            # Get response from Sukoon
            response, agent_type = self.chat(user_input, config)
            print(f"\nSukoon ({agent_type}): {response}")    

            # Store conversation turn
            turn_data = {
                "turn": i + 1,
                "timestamp": datetime.now().isoformat(),
                "student": user_input,
                "sukoon": response,
                "agent_type": agent_type,
                "conversation_id": conversation_id
            }
            
            # Update context and history
            self.current_context.append(turn_data)
            self.conversation_history.append(turn_data)
            
        # Save conversation at the end
        self.save_conversation()       
        # Optional: Add delay between turns for more natural conversation flow
        # time.sleep(1)

    def analyze_conversation(conversation_id):
        conn = sqlite3.connect('conversations.db')
        query = """
        SELECT timestamp, student, sukoon, agent_type 
        FROM conversations 
        WHERE conversation_id = ? 
        ORDER BY turn
        """
        df = pd.read_sql_query(query, conn, params=[conversation_id])
        conn.close()
        return df

if __name__ == "__main__":
    # Initialize and run conversation
    convo = ConversationManager(groq_api_key, groq_model)
    convo.run_conversation(max_turns=5)
    # convo.save_conversation(format="sqlite") # or "csv", "json"


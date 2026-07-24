# So now we are creating a graph
# First thing to make is a state
import os
from dotenv import load_dotenv

load_dotenv()

#1 typed DICT (Most Commonly used)

from typing import TypedDict

class State(TypedDict):
    topic : str 
    summary : str 
    score : int

#2 Using pydantic to create a state - It is good at data validation and type checking at runtime

from pydantic import BaseModel , field_validator

class State(BaseModel):
    topic : str 
    summary : str = ""
    score : int

    @field_validator
    def score_positive(cls , v):
        if v < 0:
            raise ValueError("Score must be positive")

# Python Dataclasses
# Standard python dataclass but not used that much

from dataclasses import dataclass , field

@dataclass
class State:
    topic : str 
    summary : str = ""
    messages : list = field(default_factory=list)

# Using Langgraph

from langgraph.graph import MessageState

class State(MessageState):
    user_name = str
    user_id = str
import os
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict

class pipelineState(TypedDict):
    raw_input = str 
    edited_text = str
    scripted_text = str
    final_output = str
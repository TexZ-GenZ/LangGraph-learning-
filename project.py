import os
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict

class pipelineState(TypedDict):
    raw_input = str 
    edited_text = str
    scripted_text = str
    final_output = str

from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(
    model = "mistral-small-2603",
    temperature = 0.7
)

def editor_node(state : pipelineState) -> dict:
    """ Stage 1 : Cleans up grammar , remove typos and refine the tone """

    prompt = (
        "You are an expert copyeditor. Clean up the following raw text ."
        "Fix any grammatical errors , spelling mistakes and smooth out the transition flow"
        "while keeping the core message intact . return only the edited text\n\n"
        f"Text: {state['raw_input']}"
    )

    response = llm.invoke(prompt)

    return {"edited_text" : response.content.strip()}

def scriptwriter_node(state : pipelineState) -> dict:
    """ Stage 2 : Formats the clean text into an engaging video style script."""

    prompt = (
        "You are a charismatic youtube content creator. Take this edited text and transform"
        "it into a highly engaging , punchy , conversational video sxript hook . Make it sound like"
        "a real person speaking passionately. return only the script content.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)

    return {"scripted_text" : response.content.strip()}

def translator_node(state : pipelineState) -> dict:
    """ Stage 3 : Translate the script into natural flowing Hinglish """

    prompt = (
        "You are an expert localizer for the Indian market . Take the following script and"
        "convert it into natural , flowing Hinglish . Do not simply translate it sentence by sentence"
        "or repeat information . Atrenating comfortably between Hindi and English just like"
        "an intellectual tech educator would speak naturally on a livee stream"
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['scripted_text']}"
    )

    response = llm.invoke(prompt)

    return {"final_output" : response.content.strip()}
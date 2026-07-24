from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph , START , END
from typing import Annotated
llm = ChatMistralAI(
    model = "mistral-small-2603",
    temperature = 0.1
)

def merge_score_dicts(existing : dict , newupdate:dict) -> dict :
    if existing is None :
        return newupdate
    return {**existing , **newupdate}

#create a state
class AnalyzerState(TypedDict):
    raw_text : str
    safety_score : Annotated[dict[str,int],merge_score_dicts]

#nodes

def toxicity_node(state : AnalyzerState) -> dict :
    print("\n [Branch 1] Analyzing Toxicity and hate speech")

    prompt = f"Analyze the following text for toxicity and hate speech:\nProvide a score from 0 to 100 where 0 is clean and 100 means highly toxic\nReturn only the plain integer number , nothing else\n\n{state['raw_text']}"

    response = llm.invoke(prompt)

    try : 
        score = int(response.content.strip())
    except ValueError :
        score = 0

    return {"safety_score" : {"toxicity" : score}}

def copyright_node(state : AnalyzerState) -> dict :
    print("\n [Branch 2] Checking Copyright")

    prompt = f"Check the following text for copyright infringement:\nProvide a score from 0 to 100 where 0 is clean and 100 means high risk\nReturn only the plain integer number , nothing else\n\n{state['raw_text']}"

    response = llm.invoke(prompt)

    try : 
        score = int(response.content.strip())
    except ValueError :
        score = 0

    return {"safety_score" : {"copyright" : score}}

def culture_node(state: AnalyzerState) -> dict :
    print("\n [Branch 3] Analyzing Cultural Sensitivity")

    prompt = f"Check the following text for cultural sensitivity:\nProvide a score from 0 to 100 where 0 is clean and 100 means highly offensive\nReturn only the plain integer number , nothing else\n\n{state['raw_text']}"

    response = llm.invoke(prompt)

    try : 
        score = int(response.content.strip())
    except ValueError :
        score = 0

    return {"safety_score" : {"sensitivity" : score}}

builder = StateGraph(AnalyzerState)

builder.add_node("toxicity" , toxicity_node)
builder.add_node("copyright" , copyright_node)
builder.add_node("culture" , culture_node)

builder.add_edge(START , "toxicity")
builder.add_edge(START , "copyright")
builder.add_edge(START , "culture")
builder.add_edge("toxicity" , END)
builder.add_edge("copyright" , END)
builder.add_edge("culture" , END)

app = builder.compile()

sample_script = """ Barcelona is the worst team in the world lol """
initial_state = {
    "raw_text" : sample_script,
    "safety_score" : {}
}

final_state = app.invoke(initial_state)

print(final_state["safety_score"])
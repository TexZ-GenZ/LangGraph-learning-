import os 
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict , Annotated , Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph , START , END
from langgraph.prebuilt import ToolNode
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt , Command



search_tool = TavilySearch(
    max_results = 3
)

tool = [search_tool]

writer_llm = ChatMistralAI(
    model = "mistral-small-2603",
    temperature = 0.7
)
writer_llm_with_tools =  writer_llm.bind_tools(tool)

reviewer_llm = ChatMistralAI(
    model = "mistral-large-2512",
    temperature = 0.1
)

class State(TypedDict) :
    topic : str 
    messages : Annotated[list , add_messages]
    draft : str
    reviewer_feedback : str 
    is_approved : bool 
    attempt : int 

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)

def writer_node(state : State) -> dict :
    """Writes (or rewrites) the LinkedIn post. can call tavily to search first."""

    topic = state['topic']
    previous_feedback = state['reviewer_feedback']

    if not previous_feedback:
        user_message = (
            f"Write a LinkedIn post on this topic {topic}"
            f"if you need current info search the web first "
        )
    else:
        user_message = (
            f"your previous draft on '{topic}' was rejected"
            f"Here is the reviewer's feedback \n\n {previous_feedback}\n\n"
            f"Write a new, improved draft that fixes every issue mentiond"
            f"do not repeat the same mistake"
        )

    messages = [("system", WRITER_SYSTEM_PROMPT)] + state["messages"] + [("human", user_message)]

    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages" : [("human",user_message),response],
    }


tool_node = ToolNode(tool)

def extract_draft_node(state : State) -> dict :
    """Extracts the draft from the state and returns it as a string."""
    last_message = state["messages"][-1]
    draft = last_message.content.strip()
    print("\n\n generated post \n\n" , draft)
    return {"draft" : draft}



def human_review_node(state : State) -> dict :
    """Pauses the graph and waits for human to approve"""

    attempt = state.get("attempt" , 0) + 1

    human_response = interrupt({
        "draft" : state["draft"],
        "attempt" : state["attempt"],
        "instructions" : "type 'approved' to accept, or type your feedback to request a rewrite"
    })

    response = human_response.strip()

    if response.lower() in ["approved" , "approve" , "yes" , "ok" , "good"] :
        return {
            "is_approved" : True,
            "reviewer_feedback" : "Approved by human."
        }
    else :
        return {
            "is_approved" : False,
            "reviewer_feedback" : response,
            "attempt" : attempt
        }

# Router Function 

def should_use_tool(state : State) : 
    last_message = state["messages"][-1]

    if getattr(last_message,'tool_calls',None):
        return "tools"
    return "extract_draft"

def should_stop_looping(state : State) : 
    if state['is_approved'] :
        print("\n\n Post Approved \n\n")
        return END
    if state['attempt'] >= 3 :
        print("Reached Max Attempts")
        return END

    return "writer"

# Build the Graph 

graph = StateGraph(State)

graph.add_node("writer" , writer_node)
graph.add_node("tools" , tool_node)
graph.add_node("extract_draft" , extract_draft_node)
graph.add_node("reviewer" , human_review_node)

graph.add_edge(START , "writer")

graph.add_conditional_edges(
    "writer" , should_use_tool
)

graph.add_edge("tools", "writer") 
graph.add_edge("extract_draft" , "reviewer")

graph.add_conditional_edges(
    "reviewer" , should_stop_looping
)

checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)

print("=" * 55)
print("Welcome to the LinkedIn Post Generator")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, review it")
print("yourself, and iterate until it's publish-ready.")

print("=" * 55)

topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

if not topic:
    print("\nNo topic given. Exiting.")
else:
    print("\nStarting generation...\n")

    config = {"configurable" : {"thread_id":"linkedin_session_1"}}

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "reviewer_feedback": "",
        "is_approved": False,
        "attempt": 1,
    }

    result = app.invoke(initial_state , config = config)

    while "__interrupt__" in result :
        interrupt_data = result["__interrupt__"][0].value

        print("\n" + "=" * 55)
        print(f"DRAFT FOR YOUR REVIEW (Attempt {interrupt_data['attempt']})")
        print("=" * 55)
        print(interrupt_data["draft"])
        print("=" * 55)
        print(f"\n{interrupt_data['instructions']}\n")

        human_input = input("\n Your response: ").strip()

        result = app.invoke(Command(resume=human_input) , config = config)


    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(result["draft"])
    print("=" * 55)
    print(f"Total attempts: {result['attempt']}")
    print(f"Approved: {result['is_approved']}")
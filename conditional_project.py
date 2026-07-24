from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict , Annotated , Literal
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph , START , END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from rich import print

#Step 1 - Building RAG retriever

embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
)
 
def build_retriever(pdf_path : str) :
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    return vectorstore.as_retriever(search_kwargs={"k": 4})

academic_retriever = build_retriever("academics_handbook.pdf")
fee_retriever = build_retriever("fee_structure.pdf")

# Step 2 - LLM

llm = ChatMistralAI(
    model = "mistral-small-2603",
    temperature = 0.4
)

# Step 3 - Define the graph

class State(TypedDict):
    programme : str 
    messages : Annotated[list, add_messages]
    query_type : str
    retrieved_context : str 


def classifier_node(state : State) -> dict :
    """Look at the latest user message and decide which path to take."""

    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)

    category = response.content.strip().lower()

    if 'academic' in category :
        category = "academic"
    elif 'fee' in category :
        category = "fee"
    else :
        category = "general"

    return {"query_type" : category}


def academic_rag_node(state : State) -> dict :
    """Retrieves relevant chunks from the academics handbook."""

    query = state["messages"][-1].content

    docs = academic_retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    return {"retrieved_context" : context}


def fee_rag_node(state : State) -> dict :
    """Retrieves relevant chunks from the fee structure."""

    query = state["messages"][-1].content

    docs = fee_retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    return {"retrieved_context" : context}

def general_node(state : State) -> dict :
    """Answer directly using the llm."""

    return {"retrieved_context" : "NO_RETRIEVAL_NEEDED"}

def response_node(state: State) -> dict :
    """Generates the final response , personalized using the student's programme."""

    query = state["messages"][-1].content

    programme = state.get("programme","Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}

# Router 

def route_query(state : State) -> Literal['academic_rag', 'fee_rag', 'general'] :
    if state["query_type"] == "academic" :
        return "academic_rag"
    elif state["query_type"] == "fee" :
        return "fee_rag"
    else :
        return "general"

# Build the graph 

graph = StateGraph(State)

graph.add_node("classifier" , classifier_node)
graph.add_node("academic_rag" , academic_rag_node)
graph.add_node("fee_rag" , fee_rag_node)
graph.add_node("general" , general_node)
graph.add_node("response" , response_node)

# edges 

graph.add_edge(START , "classifier")

graph.add_conditional_edges(
    "classifier", route_query
)

graph.add_edge("academic_rag" , "response")
graph.add_edge("fee_rag" , "response")
graph.add_edge("general" , "response")

graph.add_edge("response" , END)

app = graph.compile()

# Run the app

print("Welcome to College Assistant\n\n")

print("Which programme are you in ")
print("1. BCA")
print("2. BBA")
print("3. B.com (H)")

choice = input("Enter your choice (1/2/3): ")

programme_map = {
    '1' : "BCA",
    '2' : "BBA",
    '3' : "B.com (H)"
}

programme = programme_map.get(choice , "BCA")

print(f"\nGreat! You're set as a {programme} student\n")

while True :
    user_input = input("You: ")

    if user_input.lower() in ["exit","quit","q"] :
        break

    result = app.invoke({
        "messages" : [("human", user_input)] , 
        "programme" : programme
    })

    print("\nAI: " + result["messages"][-1].content)
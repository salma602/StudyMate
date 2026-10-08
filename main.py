from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph.message import add_messages


load_dotenv()  
llm=ChatGroq(
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b"
)
#response = llm.invoke("Explain what an AI agent is to a beginner.")
#print(response.content)


#sate graph
class StudyMateState(TypedDict):
    answer: str
    user_question: str   
    is_correct: bool
    max_retries: int
    retry_count: int
    messages: Annotated[list, add_messages]

#first node
def generate_answer(state: StudyMateState) :
    question = state.get("user_question", "")
    response = llm.invoke(question)
    return {
        "answer": response.content,
        "user_question": question
    }

def format_answer(state: StudyMateState):
    answer = state.get("answer", "")
    formatted_answer = f"Answer: {answer}"
    return {
        "answer": formatted_answer,
    }
def check_answer(state: StudyMateState):
    answer = state.get("answer", "")
    question = state.get("user_question", "")
    if not answer or not question:
        return {
            "answer": answer,
            "is_correct": False,
            "retry_count": state.get("retry_count", 0) + 1
        }
    else:
        return {
            "answer": answer,
            "is_correct": True
        }


def decide_next_step(state: StudyMateState):

    if state.get("is_correct", False):
        return "end"
    else:
        if state.get("retry_count", 0) >= state.get("max_retries", 3):
            return "end"
        else:
            return "retry"


@tool
def calculator(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

llm_with_tools = llm.bind_tools([calculator, multiply])
tools = {
    "calculator": calculator,
    "multiply": multiply
}



def call_llm(state:StudyMateState):
    response=llm_with_tools.invoke(state["messages"])
    return {
        "messages":[response]
    }

def should_continue(state: StudyMateState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"
    else:
        return "end"

def tool_node(state: StudyMateState):
    last_message = state["messages"][-1]

    tool_messages = []

    for tool_call in last_message.tool_calls:

        if tool_call["name"] not in tools:
            raise ValueError(f"Tool {tool_call['name']} not found.")

        tool = tools[tool_call["name"]]
        result = tool.invoke(tool_call["args"])

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    return {
        "messages": tool_messages
    }

graph_builder=StateGraph(StudyMateState)
graph_builder.add_node("generate_answer",generate_answer)
graph_builder.add_edge(START,"generate_answer")

graph_builder.add_node("format_answer",format_answer)

graph_builder.add_edge("generate_answer", "format_answer")
graph_builder.add_node("check_answer",check_answer)
graph_builder.add_edge("format_answer", "check_answer")

graph_builder.add_conditional_edges("check_answer",decide_next_step, {
    "end": END,"retry": "generate_answer"})
graph = graph_builder.compile()


result=graph.invoke({
    "user_question": "",
    "answer": "",
    "max_retries": 3,
    "retry_count": 0
      })
print(result["answer"])
print(result["is_correct"])
print(result["retry_count"])



#Tool graph

tool_graph_builder = StateGraph(StudyMateState)

tool_graph_builder.add_node("call_llm", call_llm)
tool_graph_builder.add_node("tool_node", tool_node)

tool_graph_builder.add_edge(START, "call_llm")
tool_graph_builder.add_conditional_edges(
    "call_llm",
    should_continue,
    {
        "tools": "tool_node",
        "end": END
    }
)
tool_graph_builder.add_edge("tool_node", "call_llm")
tool_graph = tool_graph_builder.compile()

result = tool_graph.invoke({
    "messages": [
        HumanMessage(
            content="What is 5 + 3, and what is 6 multiplied by 7?"
        )
    ]
})

print(result["messages"][-1].content)
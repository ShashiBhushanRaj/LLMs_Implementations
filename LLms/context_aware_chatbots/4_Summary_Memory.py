from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# Initialize model
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=llm)


# Summary Memory
store = {}
summary_store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        summary_store[session_id] = ""

    return store[session_id]


def summarize_conversation(session_id):

    history = get_session_history(session_id)

    messages = history.messages

    # Summarize older messages, retaining recent 2 turns
    old_messages = messages[:-4]

    if not old_messages:
        return

    summary_prompt = f"""
    Update the following conversation summary using the new messages.

    Previous Summary:
    {summary_store[session_id]}

    New Conversation:
    {old_messages}

    Create a concise summary preserving important facts,
    user preferences, names and context.
    """

    response = chat_model.invoke(summary_prompt)

    summary_store[session_id] = response.content

    # Remove summarized messages from history
    history.clear()

    history.add_messages(messages[-4:])


def chat(session_id, user_input):

    history = get_session_history(session_id)

    # Summarize when conversation becomes long
    if len(history.messages) >= 6:
        summarize_conversation(session_id)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful AI assistant.

        Previous Conversation Summary:
        {summary}
        """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | chat_model

    response = chain.invoke({
        "summary": summary_store[session_id],
        "history": history.messages,
        "input": user_input
    })

    # Save current conversation
    history.add_user_message(user_input)
    history.add_ai_message(response.content)

    return response.content


# First interaction
resp1 = chat("1", "Hi, my name is Shashi")
print("AI:", resp1)

# Second interaction
resp2 = chat("1", "I am studying Electronics and Communication Engineering")
print("AI:", resp2)

# Third interaction
resp3 = chat("1", "I am interested in AI and healthcare")
print("AI:", resp3)

# Fourth interaction
resp4 = chat("1", "What is my name and branch?")
print("AI:", resp4)
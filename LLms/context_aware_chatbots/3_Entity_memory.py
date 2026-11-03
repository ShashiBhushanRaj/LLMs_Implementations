
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# Initialize model
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=llm)

# Entity Memory
entity_store = {}

def extract_entities(text):

    prompt = ChatPromptTemplate.from_template("""
    Extract important entities and their facts from this conversation.

    Return the result in this format:
    Entity: Information

    Conversation:
    {text}
    """)

    chain = prompt | chat_model | StrOutputParser()

    return chain.invoke({"text": text})


def chat(session_id, user_input):

    if session_id not in entity_store:
        entity_store[session_id] = {}

    memory = entity_store[session_id]

    # Retrieve existing entity information
    existing_memory = "\n".join(
        f"{entity}: {info}"
        for entity, info in memory.items()
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant.

    Relevant Entity Memory:
    {memory}

    User Question:
    {question}

    Answer using the available entity information when relevant.
    """)

    chain = prompt | chat_model | StrOutputParser()

    response = chain.invoke({
        "memory": existing_memory,
        "question": user_input
    })

    # Extract entities from the new interaction
    extracted = extract_entities(
        f"User: {user_input}\nAssistant: {response}"
    )

    # Update entity store
    for line in extracted.splitlines():

        if ":" in line:
            entity, info = line.split(":", 1)

            entity = entity.strip().lstrip("-* ")
            info = info.strip()

            if entity and info:
                memory[entity] = info

    return response


# First interaction
print(chat("1", "Hi, my name is Shashi."))

# Second interaction
print(chat("1", "I study Electronics and Communication Engineering at IIIT Una."))

# Third interaction
print(chat("1", "I am interested in healthcare AI."))

# Fourth interaction
print(chat("1", "What do you know about my education?"))

# Display stored entities
print("\nENTITY MEMORY:")
print(entity_store["1"])

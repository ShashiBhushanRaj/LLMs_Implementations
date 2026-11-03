from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline ,ChatHuggingFace

llm=HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation"
)

chat_model=ChatHuggingFace(llm=llm)

##Buffer Memory
store={ } #maintaining chat history outside the chain

def get_session_history(session_id:str)->InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id]=InMemoryChatMessageHistory()
    return store[session_id]

chain=RunnableWithMessageHistory(chat_model,get_session_history)

resp1=chain.invoke(
    "hi, my name is Shashi",
    config={"configurable":{"session_id":"1"}}
)

print(resp1.content)

resp2=chain.invoke(
    "what is  my name",
    config={"configurable":{"session_id":"1"}}
)

print(resp2.content)
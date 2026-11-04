import os
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline


llm=HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task='text-generation'
)

chat_model=ChatHuggingFace(llm=llm)

msg1="Hello! my name is Shashi.who are you? "
resp1=chat_model.invoke(msg1)
print(f'-{msg1}\n{resp1.content}')

msg2="What is my name?"
resp2=chat_model.invoke(msg2)
print(f'-{msg2}\n{resp2.content}')
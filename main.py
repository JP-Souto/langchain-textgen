import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "false"
from transformers import pipeline # pipeline is a simplified way of running various models
from langchain_huggingface import HuggingFacePipeline # allows wrap a HuggingFace model as a langchain model
from langchain.prompts import PromptTemplate
import torch

# checks if gpu is available
print(torch.cuda.is_available()) #should print True
print(torch.cuda.get_device_name(0)) #should print GPU name

# pipeline sets everything up with (task, model, device = cpu or "0"(gpu))
model = pipeline("text-generation",
                 model="alamios/Mistral-Small-3.1-DRAFT-0.5B",
                 device=0,
                 max_length=256,
                 truncation=True,
                 )

# create the llm by wrapping the model in a HuggingFace pipeline
llm = HuggingFacePipeline(pipeline=model)

template = PromptTemplate.from_template("Explain {topic} for a {age} year old to understand.")

# creates the langchain chain and fills it with the correct variables to pass to the llm
chain = template | llm
topic = input("topic: ")
age = input("age: ")

# the arguments are passed as a dictionary due to llms
response = chain.invoke({"topic": topic, "age": age})
print(response)

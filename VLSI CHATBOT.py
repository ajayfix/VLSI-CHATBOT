
# SimpleDirectoryReader is used to read the given data base i.e feed DB into GPT
# GPTListIndex is used to index the fed data
# GPTSimpleVectorIndex is used to load our indexed data
# LLMPredictor is used to predict the Large Language Model.

# GPT Trained with our data

# Libraries Requried :
# 1. pip install gpt_index
# 2. pip install langchain

from gpt_index import SimpleDirectoryReader,GPTListIndex,GPTSimpleVectorIndex,LLMPredictor,PromptHelper,ServiceContext
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
import os
import gradio

os.environ["OPENAI_API_KEY"] = "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"

def create_index(directory_path):
  maximum_input = 4096
  out_tokens = 300
  chunk_size = 1200  # for LLM, we need to define chunk size
  maximum_chunk_overlap = 40

  # PromptHelper is used to assign the max token limit,max input limit
  prompthelper = PromptHelper(maximum_input, out_tokens, maximum_chunk_overlap, chunk_size_limit=chunk_size)

  # LLM- In OpenAI there are different types of Models available using LLM we can assign any model from it.
  # llmPredictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=out_tokens))
  llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-003", max_tokens=out_tokens))

  # load data — SimpleDirectoryReader is used to load the data from the directory_path to data_files list
  data_files = SimpleDirectoryReader(directory_path).load_data()
  print(directory_path)

  # create vector index
  service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor, prompt_helper=prompthelper)
  vectorIndex = GPTSimpleVectorIndex.from_documents(documents=data_files, service_context=service_context)
  vectorIndex.save_to_disk("vectorIndexfile.json")
  return vectorIndex

def query_function(Query):
  vIndex = GPTSimpleVectorIndex.load_from_disk("vectorIndexfile.json")
  query_prompt = Query
  response = vIndex.query(query_prompt, response_mode="compact")
  return response

vectorIndex = create_index("C:/Users/kotturi_ganesh/PycharmProjects/API_CHATBOT/TRANNING_DATA_SET")

#Creating the UI using Gradio Library
interface = gradio.Interface(
  title="VLSI - ChatBot",
  fn=query_function,
  inputs=gradio.components.Textbox(lines= 2,placeholder= "Enter you question here..."),
  outputs=gradio.components.Textbox(lines = 30,placeholder=" "),
)

#Launching the Gradio interface here
interface.launch(debug = True)
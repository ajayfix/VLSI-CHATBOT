# VLSI-CHATBOT

Install following Packages/Libraries:
1. gpt_index
2. langchain
3. os
4. gradio

command to install packages : pip install <package_name>

from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext

from langchain import OpenAI, DirectoryLoader, ChatOpenAI

# Using create_index add the directory path to the Training/Custom Data.
# Which is passed to the function create_index()

We used SimpleDirectoryReader to define the path of the data that needs to be loaded into the program.

We used GPTListIndex we split the data set into a list.

Now to convert this list data into vectors we used GPTSimpleVectorIndex.

We used LLMPredictor to define the AI model. The model that we used here is Text-ada-003, a pre-trained model.

Using PromptHelper we pass the parameters like maximum token size, maximum chunk overlap, maximum input.

Using ServiceContext function we passed the LLMpredictor and PromptHelper along with the training data to the vectorIndex function.

Thus we created a JSON file to store the converted vector data in the local host.

Using Scemantic Search method the model searches the relevent answers to a given prompt.

We built a simple interface using gradio library and passed query_function to it.

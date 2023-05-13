import sys
import llamacpp
from typing import Optional, List, Mapping, Any
import gradio as gr
from langchain.llms.base import LLM
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain, LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain, LLMBashChain
################################# Agentes ###################################
from langchain.prompts import StringPromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
import re

################################# CHAT ########################################
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

window_memory = ConversationBufferWindowMemory(k=4)
PATH = "../modelos"

Model_max_length= 2048

def progress_callback(progress):
    print("Progress: {:.2f}%".format(progress * 100))
    sys.stdout.flush()


params = llamacpp.InferenceParams.default_with_callback(progress_callback)
params.path_model = '/mnt/hdd/IA/Txt2Txt/Oobabooba/text-generation-webui/models/ggml-vicuna-13b-4bit.bin'

params.n_ctx = Model_max_length
params.temp = 0.1
params.n_threads = 4
model = llamacpp.LlamaInference(params)

class CustomLLM(LLM):

    def check_repeated_phrases(self,text):
        words = text.split()  # split the text into words
        phrase_counts = {}  # create a dictionary to store the counts of phrases
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                phrase = ' '.join(words[i:j])  # create a phrase from the words
                if len(phrase.split()) > 10:  # check if the phrase has more than 10 words
                    if phrase in phrase_counts:
                        phrase_counts[phrase] += 1
                        if phrase_counts[phrase] > 1:
                            return True  # if the phrase is repeated more than once, return True
                    else:
                        phrase_counts[phrase] = 1
        return False  # if no repeated phrases with more than 10 words are found, return False


    def remove_question_lines(self, text):
        lines = text.split('\n')  # split the text into individual lines
        filtered_lines = [line for line in lines if 'Question:' not in line]  # remove any line that contains the word "Question"
        return '\n'.join(filtered_lines)  
     
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_tokens = model.tokenize(prompt, True)
        model.update_input(prompt_tokens)
        model.ingest_all_pending_input()
        response = ""
        for i in range(Model_max_length):
            if response.find("end of answer") != -1:
                break
            if response.find("Observation:") != -1:
                break
            if self.check_repeated_phrases(response):
                break
            model.eval()
            token = model.sample()
            text = model.token_to_str(token)
            sys.stdout.write(text)
            sys.stdout.flush()
            response = response + text

        #remove the entire line that contains "Question:"
        response = self.remove_question_lines(response)
        return response


   

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"Vicuna": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"





local_llm = CustomLLM()

#get devices in pytorch


    

# Define which tools the agent can use to answer user queries
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name='Search',
        func= search.run,
        description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
    )
]

###### Custom Output Parser
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        lines = llm_output.split('\n')  # split the text into individual lines
        action = [line for line in lines if 'Function:' in line][-1].split(':')[1]
        action_input = [line for line in lines if 'Parameter:' in line][-1].split(':')[1]

        # Return the action and action input
        return AgentAction(tool=action.strip(" ").strip('"'), tool_input=action_input.strip(" ").strip('"'), log=llm_output)

   

output_parser = CustomOutputParser()

###### Custom Prompt
# Set up the base template
template = """Answer the following questions as best you can. You have access to the following functions:

{tools}

Use the following format strictly, don't change the fields names:

Question: the input question you must answer
Thought: you should always think about what to do
Function: the function to execute, should be one of these [{tool_names}] 
Parameter: the parameter to the function in order to answer the question
Observation: the result of the fuction

Pay special atention to the observation field.
If you there is not an answer to the question do the "Thought, Function, Parameter, Observation" sequence again.
If there is a answer to the original question stop and write the final answer 

Final Answer: the final answer to the original input question and stop


### Instruction:
 {input}
{agent_scratchpad}
### Assistant:
"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

#### Agent

llm_chain = LLMChain(llm=local_llm, prompt=prompt)

tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["Observation:"], 
    allowed_tools=tool_names
)

def evaluate(
    instruction
):
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    response = agent_executor.run({"input":instruction})
    return response


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def delete_history(history):
    history = history.clear()
    return history

def bot(history):
    history[-1][1] = evaluate(history[-1][0])
    return history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.Button("Borrar Historial")

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )
    btn.click(delete_history, [chatbot, btn], [chatbot])

demo.launch()

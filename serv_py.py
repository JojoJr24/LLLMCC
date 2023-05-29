import os
os.environ['TRANSFORMERS_CACHE'] = '/mnt/hdd/IA/Txt2Txt/modelos'
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch

import sys
from typing import Optional, List, Mapping, Any
import flask
from langchain.llms.base import LLM

Model_max_length= 4000
Max_words_per_call = 4000

checkpoint = "bigcode/tiny_starcoder_py"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint,device_map="auto")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=Model_max_length,
    temperature=0,
    top_p=0.95,
    repetition_penalty= 1.2 
)


class CustomLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = pipe(prompt)[0]["generated_text"]
        return response.replace(prompt,"")



   

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"Tiny": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"





local_llm = CustomLLM()



app = flask.Flask(__name__)

@app.route("/")
def index():
  return "Hello World!"

@app.route("/addCode", methods=["POST"])
def run():
    prompt = flask.request.json["prompt"]
    lenguaje = flask.request.json["lenguaje"]
    proyecto = flask.request.json["proyecto"]  
    p_arr = prompt.split(" ")
    if len(p_arr)>Max_words_per_call:
        p_arr = p_arr[-Max_words_per_call:]
        prompt = " ".join(p_arr)

    context = prompt
    print(prompt)
    response = local_llm(context)
    return flask.jsonify({"response": response})

@app.route("/debugCode", methods=["POST"])
def debug():
    code = flask.request.json["code"]
    lenguaje = flask.request.json[ "lenguaje"]
    proyecto = flask.request.json["proyecto"]  

    context = f"""Human will give you code and comments of code. The Language is {lenguaje}. 

    ### Human:
    Why this code doesn't work?
    {code}
    ### Assistant:"""
    
    response = local_llm(context)
    return flask.jsonify({"response": response})

if __name__ == "__main__":
  app.run(debug=True)
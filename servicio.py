import sys
import llamacpp
from typing import Optional, List, Mapping, Any
import flask
from langchain.llms.base import LLM

Model_max_length= 100
Max_words_per_call = 100

def progress_callback(progress):
    print("Progress: {:.2f}%".format(progress * 100))
    sys.stdout.flush()


params = llamacpp.InferenceParams.default_with_callback(progress_callback)
params.path_model = '/mnt/hdd/IA/Txt2Txt/Oobabooba/text-generation-webui/models/ggml-vicuna-13b-4bit.bin'

params.n_ctx = Model_max_length
params.temp = 0.7
model = llamacpp.LlamaInference(params)

class CustomLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_tokens = model.tokenize(prompt, True)
        model.update_input(prompt_tokens)
        model.ingest_all_pending_input()
        response = ""
        for i in range(Model_max_length):
            if response.find("### Human") != -1:
                break
            if response.count("\n") > 4:
                break
            model.eval()
            token = model.sample()
            text = model.token_to_str(token)
            sys.stdout.write(text)
            sys.stdout.flush()
            response = response + text
        return response


   

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"Vicuna": self.model_name}

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
    lenguaje = flask.request.json[ "lenguaje"]
    proyecto = flask.request.json["proyecto"]  
    p_arr = prompt.split(" ")
    if len(p_arr)>Max_words_per_call:
        p_arr = p_arr[-Max_words_per_call:]
        prompt = " ".join(p_arr)

    context = f"""Human will give you code and comments of code in {lenguaje}. Read it and write the part that follows  
    ### Human:
    {prompt}
    ### Assistant:"""
    
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
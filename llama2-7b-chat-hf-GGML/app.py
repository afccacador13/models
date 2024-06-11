import os
import gradio as gr
import ctransformers

configObj = ctransformers.Config(stop=["\n", 'User'], context_length=2048)
config = ctransformers.AutoConfig(config=configObj, model_type='llama')
config.config.stop = ["\n"]

# path_to_llm = os.path.abspath("llama-2-7b-chat.ggmlv3.q4_1.bin")

llm = ctransformers.AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q4_K_M.gguf", config=config)

def complete(prompt, stop=["User", "Assistant"]):
  tokens = llm.tokenize(prompt)
  output = ''
  for token in llm.generate(tokens):
    result = llm.detokenize(token)
    output += result
    for word in stop:
      if word in output:
        print('\n')
        return output
    print(result, end='',flush=True)

  print('\n')
  return output

title = "llama2-7b-chat-ggml"
description = "This space is an attempt to run the GGUF 4 bit quantized version of 'llama2-7b-chat' on a CPU"

example_1 = "Write a 7 line poem on AI"
example_2 = "Tell me a joke"

examples = [example_1, example_2]

def generate_response(user_input):
    prompt = f'User: {user_input}\nAssistant: '
    response = complete(prompt)
    return response

UI = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="User Query", placeholder="Ask your queries here...."),
    outputs=gr.Textbox(label="Assistant Response"),
    title=title,
    description=description,
    examples=examples
)

UI.launch()
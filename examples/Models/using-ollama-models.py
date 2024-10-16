
""" This example illustrates how to use Ollama models in llmware.  It assumes that you have separately
    downloaded and installed Ollama and used 'ollama run {model_name}' to cache several models in
    ollama. """

from llmware.models import ModelCatalog

#   Step 1 - register your Ollama models in llmware ModelCatalog
#   -- these two lines will register: llama2 and mistral models
#   -- note: assumes that you have previously cached and installed both of these models with ollama locally

model_name = "qwen2.5:0.5b"

#   register model
ModelCatalog().register_ollama_model(model_name=model_name,model_type="chat",host="localhost",port=11434)

#   optional - confirm that model was registered
my_new_model_card = ModelCatalog().lookup_model_card(model_name)
print("\nupdate: confirming - new ollama " + model_name + " model card - ", my_new_model_card)

#   Step 2 - start using the Ollama model like any other model in llmware

print("\nupdate: calling ollama model ...")

model = ModelCatalog().load_model(model_name)
response = model.inference("why is the sky blue?")

print("update: example #1 - ollama " + model_name + " response - ", response)

context_passage= ("NASA’s rover Perseverance has gathered data confirming the existence of ancient lake "
                  "sediments deposited by water that once filled a giant basin on Mars called Jerezo Crater, "
                  "according to a study published on Friday. The findings from ground-penetrating radar "
                  "observations conducted by the robotic rover substantiate previous orbital imagery and "
                  "other data leading scientists to theorize that portions of Mars were once covered in water "
                  "and may have harbored microbial life.  The research, led by teams from the University of "
                  "California at Los Angeles (UCLA) and the University of Oslo, was published in the "
                  "journal Science Advances. It was based on subsurface scans taken by the car-sized, six-wheeled "
                  "rover over several months of 2022 as it made its way across the Martian surface from the "
                  "crater floor onto an adjacent expanse of braided, sedimentary-like features resembling, "
                  "from orbit, the river deltas found on Earth.")

response = model.inference("What are the top 3 points?", add_context=context_passage)

print("\nupdate: calling ollama " + model_name + " model ...")

print("update: example #2 - ollama " + model_name + " response - ", response)

#   Step 3 - using the ollama discovery API - optional

discovery = model.discover_models()
print("\nupdate: example #3 - checking ollama model manifest list: ", discovery)

if len(discovery) > 0:
    # note: assumes tht you have at least one model registered in ollama -otherwise, may throw error
    for i, models in enumerate(discovery["models"]):
        print("ollama models: ", i, models)


# for more information and other alternatives for using GGUF models, please see the following examples:
#   -- examples/Models/chat_gguf_fast_start.py
#   -- examples/Models/using_gguf.py
#   -- examples/Models/using-open-chat-models.py
#   -- examples/Models/dragon-gguf_fast_start.py

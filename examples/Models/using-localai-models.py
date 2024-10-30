
"""
    This example shows how to use local LLM models that expose an endpoint compatible with the
    OpenAI API - using 'api_base' to configure the endpoint uri

    For example, to integrate a model on LM Studio with standard configuration:
        -- api_base = 'http://localhost:1234/v1'

    Please also note that llmware implements llama.cpp directly, so you can run inference on any GGUF models
    very easily and natively in llmware - see the GGUF example in /Models/using_gguf.py'
"""


from llmware.models import ModelCatalog
from llmware.prompts import Prompt


#   one step process:  add the local LLM model to the Model Registry
#   key params:
#       model_name      =   "my_llm_model"
#       api_base        =   uri_path to the proposed endpoint
#       prompt_wrapper  =   alpaca | <INST> | chat_ml | hf_chat | human_bot
#                           <INST>      ->  Llama2-Chat
#                           hf_chat     ->  Zephyr-Mistral
#                           chat_ml     ->  OpenHermes - Mistral
#                           human_bot   ->  Dragon models
#       model_type      =   "chat" (alternative:  "completion")

model_name = "llama-3.2-1b-instruct:q4_k_m"

ModelCatalog().register_open_chat_model(model_name,
                                        api_base="http://192.168.1.111:8880/v1",
                                        prompt_wrapper="<INST>",
                                        model_type="chat")

#   once registered, you can invoke like any other model in llmware

prompter = Prompt().load_model(model_name)
response = prompter.prompt_main("What is the future of AI?")

print("response: ", response)

#   if you list all of the models in the catalog, you will see the two newly created open chat models
#my_models = ModelCatalog().list_all_models()

#for i, mods in enumerate(my_models):
#    print("models: ", i, mods)



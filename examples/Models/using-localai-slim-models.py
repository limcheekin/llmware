
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

model_name = "slim-sentiment-tool"
model_card = ModelCatalog().lookup_model_card(model_name)
print("# model_card:", model_card)

ModelCatalog().register_open_chat_model(model_card["gguf_file"],
                                        api_base="http://192.168.1.111:8880/v1",
                                        context_window=model_card["context_window"],
                                        instruction_following=model_card["instruction_following"],
                                        prompt_wrapper=model_card["prompt_wrapper"],
                                        temperature=model_card["temperature"],
                                        model_type="function_call")

# once registered, you can invoke like any other model in llmware
passage1 = ("This is one of the best quarters we can remember for the industrial sector "
            "with significant growth across the board in new order volume, as well as price "
            "increases in excess of inflation.  We continue to see very strong demand, especially "
            "in Asia and Europe. Accordingly, we remain bullish on the tier 1 suppliers and would "
            "be accumulating more stock on any dips.")

prompter = Prompt().load_model(model_card["gguf_file"])
# REF: https://huggingface.co/llmware/slim-sentiment-tool/blob/main/config.json
prompt_format = "<human> {context_passage} <classify> sentiment </classify>\n<bot>:"
prompt = prompt_format.format(context_passage=passage1)
print("# prompt:", prompt)
response = prompter.prompt_main(prompt)
print("# prompt response:", response['llm_response'])


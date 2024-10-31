
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
import ast

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

# add "llmware/" prefix so that it won't crash with existing one
localai_model_name = "llmware/" + model_name

# LocalAI model config at https://gist.github.com/limcheekin/be44f076cff415bd0fc64cbf3e3b3107
ModelCatalog().register_open_chat_model(localai_model_name,
                                        api_base="http://192.168.1.111:8880/v1",
                                        model_type="completion")

# once registered, you can invoke like any other model in llmware
passage1 = ("This is one of the best quarters we can remember for the industrial sector "
            "with significant growth across the board in new order volume, as well as price "
            "increases in excess of inflation.  We continue to see very strong demand, especially "
            "in Asia and Europe. Accordingly, we remain bullish on the tier 1 suppliers and would "
            "be accumulating more stock on any dips.")

prompter = Prompt().load_model(localai_model_name)
response = prompter.prompt_main(passage1)
print("# json response:", response['llm_response'])
response['llm_response'] = ast.literal_eval(response['llm_response'])
sentiment_value = response["llm_response"]["sentiment"]
print("# sentiment_value:", sentiment_value)

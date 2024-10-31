import ast

def parse(response_dict):
    response_dict['llm_response'] = ast.literal_eval(response_dict['llm_response'])
    return response_dict

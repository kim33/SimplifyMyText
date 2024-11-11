from openai import OpenAI
import argparse

def model_search(model, client):
    for model in client.models.list().data:
      model_name = model.id
      if "llama" in model_name:
        break
    return model_name    

def simplify_text(text, audience, my_key, model_type):
    if model_type == 'llama3' :
        client = OpenAI(base_url="https://llm.scads.ai/v1",api_key=my_key)
        model_name = model_search(model_type, client)
    elif model_type == 'gpt-4o':
        client = OpenAI(api_key = my_key)
        model_name = model_type

    response = client.chat.completions.create(
        model= model_name,
        messages=[
            {
                "role" : "system",
                "content" : f"You are a language expert whose goal is to simplify the text for the following audience group. Your goal is to deliver the information each line of the text is trying to convey. From the given text file, take each line, which consists of multiple sentences. Follow plain language standard, such that - use familiar words, use short words, use precise and concrete words, avoid abbreviations and filler words, form short sentences, use verbal style, use active voice, form a maximum of two subordinate clauses, use genitives sparingly. Do not add new line or indentation. Ignore any double space in line. When empty line detected, skip and move on to the next sentence. Do not change the format of the line. : {audience}"
            },
            {"role": "user",
             "content" : f"Simplify the following text. Do not make new line. Keep simplified results in the same line:{text}"
            }
        ],
        max_tokens=1500,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def simplify_file(model, input_file_path, output_file_path, audience, my_key) :
    
    with open(input_file_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    simplified_lines = []
    for line in lines:
        simplified_line = simplify_text(line.strip(), audience, my_key, model)
        simplified_lines.append(simplified_line + "\n")

    with open(output_file_path, "w", encoding="utf-8") as outfile:
        outfile.writelines(simplified_lines)

# call simplify_file
def main(args) :
    model = args.model_name
    target = args.target_user
    input_file = args.file_path
    output_file = args.result_path
    my_api_key = args.api_key
    print("moel name : ", model, "\ntarget audience: ", target, "\ntext to simplify : ", input_file, "\nresult file path : ", output_file)
    simplify_file(model, input_file, output_file, target, my_api_key)
    print("Simplification Completed")
    
    
# takes desired model, original text and result file path as arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for simplification.')
    parser.add_argument(
        '--model_name', default='gpt-4o', type=str, choices=['gpt-4o', 'llama3'],
        help='model name'
    )
    parser.add_argument(
        '--file_path', default='sample.txt', type=str,
        help='text file path to simplify'
    )
    parser.add_argument(
        '--result_path', default='./simplified_text/sample_simplified.txt', type=str,
        help='result file path'
    )
    parser.add_argument(
        '--target_user', default='dyslexia', type=str,
        help='target reader : scientists, educators, teenagers, etc.'
    )
    parser.add_argument(
        '--api_key', default='empty', type=str,
        help='OpenAI API Key'
    )
    args = parser.parse_args()
    main(args)
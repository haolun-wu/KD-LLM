import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import configs
from datasets import load_dataset
import transformers
from torch import bfloat16
import requests

# Argument parser setup
parser = argparse.ArgumentParser(description="Run NLI predictions")
parser.add_argument("--model_name", default=configs.DEFAULT_MODEL_NAME, type=str, help="Model name or path")
parser.add_argument("--dataset_path", default=configs.DEFAULT_DATASET_PATH, type=str, help="Path to the dataset")
parser.add_argument("--max_length", default=configs.DEFAULT_MAX_LENGTH, type=int,
                    help="Maximum length of the model output")
parser.add_argument("--num_beams", default=configs.DEFAULT_NUM_BEAMS, type=int, help="Number of beams for beam search")
args = parser.parse_args()

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# setup quantization config
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

# Load the model and tokenizer
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "mosaicml/mpt-7b"
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# model_name = "google/gemma-7b-it"
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)  # quantized model
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)


# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def load_validation_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def generate_prompts(data_batch):
    prompts = []
    for premise, hypothesis in zip(data_batch['premise'], data_batch['hypothesis']):
        # prompts.append(
        #     "You will be tested on the SNLI data. You should predict whether a premise entails a hypothesis or not, "
        #     "and you need to output from 0, 1, 2 three options. Namely, you are given pairs of premise and hypothesis. "
        #     "You should output 0 if the hypothesis can be entailed by the premise. You should output 1, if it is "
        #     "neutral. You should output 2, if it is contradiction.")
        prompts.append(
            f"Given the premise: '{premise}' and the hypothesis: '{hypothesis}', please first provide a one sentence "
            f"explanation on your thoughts for whether the hypothesis can be entailed by the premise, or neutral, "
            f"or contradiction. Then, output '0' if the hypothesis can be entailed by the premise, '1' if it is "
            f"neutral, otherwise output '2' if there is contradiction. Your output should use the template "
            f"'My rationale is: ... and my prediction is: 0/1/2.' in one sentence.")
    return prompts


def parse_responses_llama2(outputs):
    rationales = []
    predictions = []
    for output in outputs:
        response = output
        # print("Response:", response)
        # Assuming the model's response includes the terms 'Entailment', 'Neutral', or 'Contradiction'
        # Adjust based on your actual model output
        parts = response.split('\n')
        if len(parts) > 2 and parts[1] == '':
            rationale = parts[2]
        elif len(parts) > 1 and parts[1] != '':
            rationale = parts[1]
        else:
            if len(response.split("My rationale is: ")) > 2:
                rationale = rationale.split("My rationale is: ")[2]
            else:
                print("parts:", parts)
                rationale = "No rationale provided."
        rationales.append(rationale)

        # Determine the prediction based on the presence of keywords in the response
        if 'Entailment'.lower() in rationale.lower() or "0" in rationale.lower():
            predictions.append('0')  # 0 for Entailment
        elif 'Neutral'.lower() in rationale.lower() or "1" in rationale.lower():
            predictions.append('1')  # 1 for Neutral
        elif 'Contradiction'.lower() in rationale.lower() or "2" in rationale.lower():
            predictions.append('2')  # 2 for Contradiction
        else:
            predictions.append('unknown')  # In case none of the keywords are found

    return rationales, predictions


def parse_responses_mistral7b(outputs):
    rationales = []
    predictions = []
    for output in outputs:
        response = output
        # print("Response:", response)
        # Assuming the model's response includes the terms 'Entailment', 'Neutral', or 'Contradiction'
        # Adjust based on your actual model output
        parts = response.split('\n')
        # print("parts:", parts)
        if len(parts) == 4:
            rationale = parts[2] + parts[3]
        elif len(parts) == 3:
            rationale = parts[2]
        elif len(response.split("My rationale is: ")) == 3:
            rationale = response.split("My rationale is: ")[2]
        else:
            print("parts:", parts)
            rationale = "No rationale provided."
        rationales.append(rationale)

        # Determine the prediction based on the presence of keywords in the response
        if 'Entailment'.lower() in rationale.lower() or "0" in rationale.lower():
            predictions.append('0')  # 0 for Entailment
        elif 'Neutral'.lower() in rationale.lower().split("My prediction is: ") or "1" in rationale.lower():
            predictions.append('1')  # 1 for Neutral
        elif 'Contradiction'.lower() in rationale.lower().split("My prediction is: ") or "2" in rationale.lower():
            predictions.append('2')  # 2 for Contradiction
        else:
            predictions.append('unknown')  # In case none of the keywords are found

    return rationales, predictions


def parse_responses_phi2(outputs):
    rationales = []
    predictions = []
    for output in outputs:
        response = output
        parts = response.split('##OUTPUT')
        # print("parts:", parts)
        if len(parts) == 2:
            rationale = parts[1]
        elif len(response.split("My rationale is: ")) == 2:
            rationale = response.split("My rationale is: ")[1]
        else:
            print("parts:", parts)
            rationale = "No rationale provided."
        rationales.append(rationale)

        # Determine the prediction based on the presence of keywords in the response
        if 'Entailment'.lower() in rationale.lower() or "0" in rationale.lower():
            predictions.append('0')  # 0 for Entailment
        elif 'Neutral'.lower() in rationale.lower().split("My prediction is: ") or "1" in rationale.lower():
            predictions.append('1')  # 1 for Neutral
        elif 'Contradiction'.lower() in rationale.lower().split("My prediction is: ") or "2" in rationale.lower():
            predictions.append('2')  # 2 for Contradiction
        else:
            predictions.append('unknown')  # In case none of the keywords are found

    return rationales, predictions


def query_mistral7b(payload):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": ""}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def main():
    # validation_data = load_validation_data('data/nli/validation.json')  # Adjust the file path accordingly
    # snli_data = load_dataset(args.dataset_path)
    import json
    with open('validation_1000.json', 'r') as file:
        snli_data = json.load(file)
    validation_data = snli_data['validation']
    # print(validation_data)
    batch_size = 1  # Adjust based on your preference and system capabilities
    result = {}
    for i in range(0, len(validation_data['premise']), batch_size):
        batch = {
            'premise': validation_data['premise'][i:i + batch_size],
            'hypothesis': validation_data['hypothesis'][i:i + batch_size]
        }
        prompts = generate_prompts(batch)
        # print(f"Batch {i // batch_size + 1} prompts: {prompts}")
        # prompts = prompts[0] + prompts[1]
        # prompts = "Hello"
        prompts = prompts[0]
        print(prompts)
        # Encode the input prompt
        # input_ids = tokenizer(prompts, return_tensors="pt").input_ids.to(device)
        input_ids = tokenizer(prompts, return_tensors="pt").to(device)
        # attention_mask = tokenizer(prompts, return_tensors="pt").attention_mask.to(device)

        # Generate output using the model
        outputs = model.generate(
            **input_ids,
            # do_sample=True,
            # top_k=10,
            # num_return_sequences=1,
            # eos_token_id=tokenizer.eos_token_id,
            # pad_token_id=tokenizer.pad_token_id,
            # attention_mask=attention_mask,
            max_new_tokens=512,
        )

        # Decode the generated output
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(query_mistral7b({"inputs": prompts}))
        # outputs = query_mistral7b({"inputs": prompts})[0]["generated_text"]
        # if len(outputs.split('\n')) != 4:
        #     continue
        # print(f"Batch {i // batch_size + 1} outputs: {outputs}")
        # rationales, predictions = parse_responses_mistral7b([outputs])
        # rationales, predictions = parse_responses_llama2([outputs])
        rationales, predictions = parse_responses_phi2([outputs])
        # print(rationales, predictions)
        # exit(0)
        for premise, hypothesis, rationale, prediction, label in zip(batch['premise'], batch['hypothesis'], rationales,
                                                                     predictions,
                                                                     validation_data['label'][i:i + batch_size]):
            print(f"{i} processed.")
            print(f"\nPremise: {premise}")
            print(f"Hypothesis: {hypothesis}")
            print(f"Rationale: {rationale}")
            print(f"Prediction (0 for Entailment, 1 for Neutral, 2 for Contradiction): {prediction}")
            print(f"Ground Truth: {label}")
            result[i] = {
                'premise': premise,
                'hypothesis': hypothesis,
                'rationale': rationale,
                'prediction': prediction,
                'label': label
            }
        # break
    with open(f'result_phi2.json', 'w') as file:
        json.dump(result, file, indent=4)


if __name__ == "__main__":
    main()
    # snli_data = load_dataset(args.dataset_path)
    # validation = snli_data['validation'][:1000]
    # new_data = {"train": {}, "validation": validation, "test": {}}
    # with open("validation_1000.json", "w") as f:
    #     json.dump(new_data, f, indent=4)
    # pip install bitsandbytes accelerate

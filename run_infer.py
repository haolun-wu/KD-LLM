import json
import argparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import configs
from datasets import load_dataset
from torch import bfloat16
import requests

# Argument parser setup
parser = argparse.ArgumentParser(description="Run NLI predictions")
parser.add_argument("--model_name", default=configs.DEFAULT_MODEL_NAME, type=str, help="Model name or path")
parser.add_argument("--dataset_path", default=configs.DEFAULT_DATASET_PATH, type=str, help="Path to the dataset")
parser.add_argument("--max_length", default=configs.DEFAULT_MAX_LENGTH, type=int,
                    help="Maximum length of the model output")
parser.add_argument("--batch_size", default=configs.DEFAULT_BATCH_SIZE, type=int,
                    help="batch size")
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

# Load the pretrained model and tokenizer
if device == 'cpu':
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_config)  # quantized model
    
tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


def load_validation_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def generate_prompts(data_batch):
    prompts = []
    for premise, hypothesis in zip(data_batch['premise'], data_batch['hypothesis']):
        prompts.append(
            "You will be tested on the SNLI data. You should predict whether a premise entails a hypothesis or not, "
            "and you need to output from 0, 1, 2 three options. Namely, you are given pairs of premise and hypothesis. "
            "You should output 0 if the hypothesis can be entailed by the premise. You should output 1, if it is "
            "neutral. You should output 2, if it is contradiction. "
            f"Given the premise: '{premise}' and the hypothesis: '{hypothesis}', please first provide a one sentence "
            f"explanation on your thoughts for whether the hypothesis can be entailed by the premise, or neutral, "
            f"or contradiction. Then, output '0' if the hypothesis can be entailed by the premise, '1' if it is "
            f"neutral, otherwise output '2' if there is contradiction. Your output should use the template "
            f"'My rationale is: ... and my prediction is: 0/1/2.' in one sentence.")
    return prompts

def parse_responses_phi2(decoded_outputs):
    rationales = []
    predictions = []
    print("decoded_outputs:", len(decoded_outputs))
    for response in decoded_outputs:
        parts = response.split('##OUTPUT')
        # print("parts:", parts)
        if len(parts) == 2:
            rationale = parts[1]
        elif len(response.split("My rationale is: ")) == 2:
            rationale = response.split("My rationale is: ")[1]
        elif len(response.split("My rationale is ")) == 2:
            rationale = response.split("My rationale is ")[1]
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


def main():
    # validation_data = load_validation_data('data/nli/validation.json')  # Adjust the file path accordingly
    data_loaded = load_dataset("snli")
    validation_data = data_loaded['validation'][:4]
    batch_size = args.batch_size  # Adjust based on your preference and system capabilities

    for i in range(0, len(validation_data['premise']), batch_size):
        batch = {
            'premise': validation_data['premise'][i:i + batch_size],
            'hypothesis': validation_data['hypothesis'][i:i + batch_size]
        }
        prompts = generate_prompts(batch)
        # prompts = prompts[0] + prompts[1]
        # print("prompts:", prompts, len(prompts))
        # outputs = [generator(prompt, max_length=200, num_return_sequences=1)[0] for prompt in prompts]
        encoded_outputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        input_ids = encoded_outputs.input_ids
        attention_mask = encoded_outputs.attention_mask
        print("input_ids:", input_ids.shape, input_ids.device)
        print("attention_mask:", attention_mask.shape, attention_mask.device)

        # Generate output using the model
        outputs = model.generate(
            input_ids=input_ids,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask,
            max_new_tokens=1000,
        )
        print("outputs:", outputs.shape)

        # Decode the generated output
        # Iterate over each sequence in the tensor and decode it
        decoded_outputs = []
        for output in outputs:
            decoded_sequence = tokenizer.decode(output.tolist(), skip_special_tokens=True)
            decoded_outputs.append(decoded_sequence)
        print("decoded_outputs:", decoded_outputs)

        rationales, predictions = parse_responses_phi2(decoded_outputs)

        for premise, hypothesis, rationale, prediction, label in zip(batch['premise'], batch['hypothesis'], rationales,
                                                                     predictions,
                                                                     validation_data['label'][i:i + batch_size]):
            print(f"\nPremise: {premise}")
            print(f"Hypothesis: {hypothesis}")
            print(f"Rationale: {rationale}")
            print(f"Prediction (0 for Entailment, 1 for Neutral, 2 for Contradiction): {prediction}")
            print(f"Ground Truth: {label}")


if __name__ == "__main__":
    main()

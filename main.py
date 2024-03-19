import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import config

# Argument parser setup
parser = argparse.ArgumentParser(description="Run NLI predictions")
parser.add_argument("--model_name", default=config.DEFAULT_MODEL_NAME, type=str, help="Model name or path")
parser.add_argument("--dataset_path", default=config.DEFAULT_DATASET_PATH, type=str, help="Path to the dataset")
parser.add_argument("--max_length", default=config.DEFAULT_MAX_LENGTH, type=int,
                    help="Maximum length of the model output")
parser.add_argument("--num_beams", default=config.DEFAULT_NUM_BEAMS, type=int, help="Number of beams for beam search")
args = parser.parse_args()

# Load the pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1,
                     max_length=args.max_length, num_beams=args.num_beams)


def load_validation_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def generate_prompts(data_batch):
    prompts = []
    for premise, hypothesis in zip(data_batch['premise'], data_batch['hypothesis']):
        prompts.append(
            f"Given the premise: '{premise}' and the hypothesis: '{hypothesis}', provide a one sentence explanation on whether the hypothesis can be entailed by the premise. Then, output 'Entailment' if the hypothesis can be entailed by the premise, otherwise output 'No Entailment'.")
    return prompts


def parse_responses(outputs):
    rationales = []
    predictions = []
    for output in outputs:
        response = output['generated_text']
        # Here you might need to adjust based on how the model formats its responses
        parts = response.split('\n')
        rationales.append(parts[0] if parts else "No rationale provided.")
        predictions.append('0' if 'Entailment' in parts else '1')
    return rationales, predictions


def main():
    validation_data = load_validation_data('data/nli/validation.json')  # Adjust the file path accordingly
    batch_size = 5  # Adjust based on your preference and system capabilities

    for i in range(0, len(validation_data['premise']), batch_size):
        batch = {
            'premise': validation_data['premise'][i:i + batch_size],
            'hypothesis': validation_data['hypothesis'][i:i + batch_size]
        }
        prompts = generate_prompts(batch)
        outputs = [generator(prompt, max_length=200, num_return_sequences=1)[0] for prompt in prompts]
        rationales, predictions = parse_responses(outputs)

        for premise, hypothesis, rationale, prediction, label in zip(batch['premise'], batch['hypothesis'], rationales,
                                                                     predictions,
                                                                     validation_data['label'][i:i + batch_size]):
            print(f"\nPremise: {premise}")
            print(f"Hypothesis: {hypothesis}")
            print(f"Rationale: {rationale}")
            print(f"Prediction (0 for Entailment, 1 for No Entailment): {prediction}")
            print(f"Ground Truth: {label}")


if __name__ == "__main__":
    main()
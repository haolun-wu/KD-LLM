from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import configs
import argparse
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AdamW
import torch.nn.functional as F
from utils import generate_prompts, parse_responses_phi2
from torch import bfloat16
import requests

# Argument parser setup
parser = argparse.ArgumentParser(description="Run NLI predictions")
parser.add_argument("--model_name", default=configs.DEFAULT_MODEL_NAME, type=str, help="Model name or path")
parser.add_argument("--dataset_path", default=configs.DEFAULT_DATASET_PATH, type=str, help="Path to the dataset")
parser.add_argument("--max_length", default=configs.DEFAULT_MAX_LENGTH, type=int,
                    help="Maximum length of the model output")
parser.add_argument("--num_beams", default=configs.DEFAULT_NUM_BEAMS, type=int, help="Number of beams for beam search")
parser.add_argument("--batch_size", default=configs.DEFAULT_BATCH_SIZE, type=int, help="Batch size")
parser.add_argument("--lr", default=configs.DEFAULT_LEARNING_RATE, type=float, help="Learning rate")
parser.add_argument("--max_epoch", default=configs.DEFAULT_TRAINING_EPOCH, type=int, help="Maximum training epoch")
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



optimizer = AdamW(model.parameters(), lr=args.lr)
generator = pipeline('text-generation',
                     model=model,
                     tokenizer=tokenizer,
                     device=device,
                     max_length=args.max_length,
                     num_beams=args.num_beams)


def main():
    data_loaded = load_dataset("snli")
    # tokenized_data = snli_data.map(preprocess_function, batched=True)
    train_data = data_loaded['train'][:10]
    validation_data = data_loaded['validation'][:4]
    batch_size = args.batch_size

    # fine-tuning
    model.train()
    for epoch in range(args.max_epoch):
        for i in range(0, len(train_data['premise']), batch_size):
            batch = {
                'premise': train_data['premise'][i:i + batch_size],
                'hypothesis': train_data['hypothesis'][i:i + batch_size],
                'label': train_data['label'][i:i + batch_size]
            }
            prompts = generate_prompts(batch)
            encoded_outputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            outputs = model.generate(
                input_ids=encoded_outputs.input_ids,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=encoded_outputs.attention_mask,
                max_new_tokens=1000,
                )
            # Iterate over each sequence in the tensor and decode it
            decoded_outputs = []
            for output in outputs:
                decoded_sequence = tokenizer.decode(output.tolist(), skip_special_tokens=True)
                decoded_outputs.append(decoded_sequence)

            rationales, predictions = parse_responses_phi2(decoded_outputs)


            loss = F.cross_entropy(torch.tensor(int_predictions), batch['label'])
            print("loss:", loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # model evaluation
    model.eval()
    with torch.no_grad():
        for i in range(0, len(validation_data['premise']), batch_size):
            batch = {
                'premise': validation_data['premise'][i:i + batch_size],
                'hypothesis': validation_data['hypothesis'][i:i + batch_size]
            }
            prompts = generate_prompts(batch)
            encoded_outputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            # Generate output using the model
            outputs = model.generate(
                input_ids=encoded_outputs.input_ids,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=encoded_outputs.attention_mask,
                max_new_tokens=1000,
                )
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

    # trainer = MyTrainer(
    #     model=model,  # Your pre-trained model
    #     args=training_args,  # Training arguments
    #     train_dataset=tokenized_data['training'][:1000],  # Training dataset
    #     eval_dataset=tokenized_data['validation'][:100]  # Evaluation dataset
    # )
    # trainer.train()


if __name__ == "__main__":
    main()

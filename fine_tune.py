from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import configs
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.nn.functional as F
from utils import generate_prompts, parse_responses

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model, tokenizer, and optimizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name)
optimizer = AdamW(model.parameters(), lr=args.lr)
generator = pipeline('text-generation',
                     model=model,
                     tokenizer=tokenizer,
                     device=device,
                     max_length=args.max_length,
                     num_beams=args.num_beams)


def main():
    snli_data = load_dataset("snli")
    # tokenized_data = snli_data.map(preprocess_function, batched=True)
    train_data = snli_data['train'][:10]
    validate_data = snli_data['validation'][:10]

    # fine-tuning
    model.train()
    for epoch in range(args.max_epoch):
        for i in range(0, len(train_data['premise']), args.batch_size):
            batch = {
                'premise': train_data['premise'][i:i + args.batch_size],
                'hypothesis': train_data['hypothesis'][i:i + args.batch_size],
                'label': train_data['label'][i:i + args.batch_size]
            }
            prompts = generate_prompts(batch)
            outputs = [generator(prompt, max_length=200, num_return_sequences=1)[0] for prompt in prompts]
            rationales, predictions = parse_responses(outputs)
            print("rationales:", rationales)
            print("predictions:", predictions)
            int_predictions = [int(pred) for pred in predictions]
            print("int_predictions:", torch.tensor(int_predictions))
            print("batch['label']:", batch['label'])
            loss = F.cross_entropy(torch.tensor(int_predictions), batch['label'])
            print("loss:", loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # model evaluation
    model.eval()
    with torch.no_grad():
        for i in range(0, len(validate_data['premise']), args.batch_size):
            batch = {
                'premise': validate_data['premise'][i:i + args.batch_size],
                'hypothesis': validate_data['hypothesis'][i:i + args.batch_size]
            }
            prompts = generate_prompts(batch)
            outputs = [generator(prompt, max_length=200, num_return_sequences=1)[0] for prompt in prompts]
            rationales, predictions = parse_responses(outputs)
            for premise, hypothesis, rationale, prediction, label in zip(batch['premise'], batch['hypothesis'],
                                                                         rationales,
                                                                         predictions,
                                                                         validate_data['label'][i:i + args.batch_size]):
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

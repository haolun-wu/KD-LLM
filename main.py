import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import config  # Import configuration settings

# Function to get model's prediction for a question
def get_prediction(model, tokenizer, question):
    inputs = tokenizer.encode(question, return_tensors="pt")
    output = model.generate(inputs, max_length=config.DEFAULT_MAX_LENGTH, num_beams=config.DEFAULT_NUM_BEAMS, early_stopping=True)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Function to display question, model's answer, and ground truth
def show_question_and_answer(model, tokenizer, dataset, index):
    ground_truth = dataset['train'][index]['answer']['value']
    question = dataset['train'][index]['question']
    model_answer = get_prediction(model, tokenizer, question)
    print(f"Question {index}: {question}")
    print(f"Model's answer: {model_answer}")
    print(f"Ground truth: {ground_truth}")
    print("-----------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Generate and display model answers for questions from a specified dataset.")
    parser.add_argument('--model', type=str, default=config.DEFAULT_MODEL_NAME, help='Model name or path')
    parser.add_argument('--data', type=str, default=config.DEFAULT_DATASET_NAME, help='Dataset name')
    parser.add_argument('--config', type=str, default=config.DEFAULT_DATASET_CONFIG, help='Dataset configuration')
    parser.add_argument('--example', type=int, default=5, help='Show example')
    args = parser.parse_args()

    # Load model and tokenizer based on command line arguments
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Load the dataset
    dataset = load_dataset(args.data, args.config)

    # Display question and answers for the first 5 entries
    for index in range(args.example):  # Loop through the first 5 questions
        show_question_and_answer(model, tokenizer, dataset, index)

if __name__ == "__main__":
    main()


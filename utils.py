# def preprocess_function(examples):
#     # Tokenize the premises and hypotheses
#     return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding=True)


def generate_prompts(data_batch):
    prompts = []
    for premise, hypothesis in zip(data_batch['premise'], data_batch['hypothesis']):
        prompts.append(
            "You will be tested on the SNLI data. You should predict whether a premise entails a hypothesis or not, and you need to output from 0, 1, 2 three options. Namely, you are given pairs of premise and hypothesis. You should output 0 if the hypothesis can be entailed by the premise. You should output 1, if it is neutral. You should output 2, if it is contradiction.")
        prompts.append(
            f"Given the premise: '{premise}' and the hypothesis: '{hypothesis}', please first provide a one sentence explanation on your thoughts for whether the hypothesis can be entailed by the premise, or neutral, or contradiction. Then, output '0' if the hypothesis can be entailed by the premise, '1' if it is neutral, otherwise output '2' if there is contradiction.")
    return prompts


def parse_responses(outputs):
    rationales = []
    predictions = []
    for output in outputs:
        # response = output['generated_text']
        response = output
        # Assuming the model's response includes the terms 'Entailment', 'Neutral', or 'Contradiction'
        # Adjust based on your actual model output
        parts = response.split('\n')
        rationale = parts[0] if parts else "No rationale provided."
        rationales.append(rationale)

        # Determine the prediction based on the presence of keywords in the response
        if 'Entailment' in response:
            predictions.append('0')  # 0 for Entailment
        elif 'Neutral' in response:
            predictions.append('1')  # 1 for Neutral
        elif 'Contradiction' in response:
            predictions.append('2')  # 2 for Contradiction
        else:
            predictions.append('unknown')  # In case none of the keywords are found

    return rationales, predictions

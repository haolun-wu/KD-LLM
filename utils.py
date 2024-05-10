# def preprocess_function(examples):
#     # Tokenize the premises and hypotheses
#     return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding=True)



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

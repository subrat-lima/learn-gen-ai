from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []


def generate_response(question):
    history_string = "\n".join(conversation_history)
    inputs = tokenizer.encode_plus(history_string, question, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_token=True).strip()
    conversation_history.append(question)
    conversation_history.append(response)
    return response


def main():
    while True:
        question = input("> ")
        response = generate_response(question)
        print(response)




if __name__ == "__main__":
    main()

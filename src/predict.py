#Fine-tuned model classify one tweet sentiment

# import necessary libraries for argument parsing, file handling, and PyTorch model loading

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#find main project directory and model directory for loading the fine-tuned BERT model

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "bert-crypto-sentiment"

# Define the main function to classify the sentiment of a single tweet using the fine-tuned BERT model.

def main():
    # Parse command-line arguments for the tweet text and model directory, load the tokenizer and model,
    # and classify the sentiment of the input tweet, printing the predicted sentiment label and confidence score

    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Tweet text to classify")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    # Set the model to evaluation mode and tokenize the input tweet text, then compute the sentiment probabilities
    model.eval()
    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=1)[0]
    #select the sentiment label with the highest probability and print the predicted sentiment and confidence score
    label_id = int(torch.argmax(probabilities))
    print(f"Sentiment: {model.config.id2label[label_id]}")
    print(f"Confidence: {probabilities[label_id].item():.1%}")

# runs main function if the script is executed directly.
if __name__ == "__main__":
    main()

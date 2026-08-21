#Predict emotions for a tweet using the fine-tuned BERT model.

#import tools for reading command-line arguments, handling file paths.
import argparse
from pathlib import Path

# import PyTorch and Hugging Face Transformers libraries for loading the model.
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "bert-crypto-emotions"

# parse command-line arguments for the tweet text, threshold for emotion detection, and model directory.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Tweet text to analyse")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
# load the tokenizer and model from the specified directory, set the model to evaluation mode, and tokenize the input tweet text.

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        scores = torch.sigmoid(model(**inputs).logits)[0]
    # print the emotion scores for each of the five emotions, and identify which emotions are detected above the specified threshold.

    print("Emotion scores:")
    detected = []
    for index, score in enumerate(scores):
        emotion = model.config.id2label[index]
        print(f"  {emotion}: {score.item():.1%}")
        if score.item() >= args.threshold:
            detected.append(emotion)
    print("Detected emotions:", ", ".join(detected) if detected else "none above threshold")

# runs the main function if the script is executed directly.

if __name__ == "__main__":
    main()

"""
Evaluates the Project 1 indicator + MLP system on the same train/validation
split used in Project 2.
"""

from pathlib import Path
import pandas as pd
from datasets import load_dataset
from tokenization import train_tokenizer
from representations import TokenIndicator
from training import (
    train_neural_network_model,
    predict,
    score_predictions,
)

# Use the official splits like in Project 2.
dataset = load_dataset("cardiffnlp/tweet_eval", "emotion")

training_texts = list(dataset["train"]["text"])
training_set = list(zip(
    training_texts,
    dataset["train"]["label"],
))
validation_texts = list(dataset["validation"]["text"])
validation_labels = list(dataset["validation"]["label"])
num_classes = len(dataset["train"].features["label"].names)

# Learn the BPE vocabulary from training texts only.
# Share this tokenizer across the three model seeds.
tokenizer = train_tokenizer(
    training_texts,
    vocab_size=1000,
    min_frequency=2,
)
encoder = TokenIndicator(tokenizer)

results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True)
results = []

for seed in [0, 1, 2]:
    print(f"\nProject 1 indicator + MLP — seed {seed}", flush=True)

    # Keep the original Project 1 architecture and training settings.
    model, history = train_neural_network_model(
        training_set,
        encoder,
        num_classes,
        model_option="mlp",
        hidden_dim=128,
        dropout=0.2,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=20,
        seed=seed,
        device="cpu",
        verbose=False,
    )

    predictions = predict(validation_texts, model, encoder)
    scores = score_predictions(
        validation_labels,
        predictions,
        num_classes,
    )

    results.append({
        "experiment": "Project 1: indicator + MLP",
        "seed": seed,
        "train_size": len(training_set),
        "validation_size": len(validation_texts),
        "num_epochs": 20,
        "actual_vocab_size": len(encoder),
        **scores,
    })

    # Save each completed run.
    pd.DataFrame(results).to_csv(
        results_dir / "project1_comparison.csv",
        index=False,
    )

    print(
        f"Accuracy: {scores['accuracy']:.3f} | "
        f"Macro-F1: {scores['macro_f1']:.3f}"
    )

# Summarise variation across the three model seeds.
summary = pd.DataFrame(results).groupby("experiment").agg(
    accuracy_mean=("accuracy", "mean"),
    accuracy_std=("accuracy", "std"),
    macro_f1_mean=("macro_f1", "mean"),
    macro_f1_std=("macro_f1", "std"),
)

print("\nProject 1 comparison results")
print(summary.round(3))

summary.to_csv(results_dir / "project1_comparison_summary.csv")

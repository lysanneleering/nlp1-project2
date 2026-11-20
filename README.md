# NLP1 Project 2

Emotion classification on the **emotion** subset of [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval): anger, joy, optimism and sadness. Building on Project 1, we replace sparse text representations with pretrained [GloVe Twitter](https://nlp.stanford.edu/projects/glove/) embeddings (25d) and compare an embed-and-pool encoder (mean/sum), an RNN and an LSTM.

## Experiments

All models are trained on the official training split and evaluated on the validation split (374 tweets); the test split is not used. Everything is in `project2.ipynb`:

- **Embeddings:** tokenizer choice (`TweetTokenizer` vs. NLTK `word_tokenize`), vocabulary coverage and out-of-vocabulary tokens, nearest neighbours, cosine similarities and analogies for the four emotion words.
- **Encoders:** mean pool, sum pool, RNN and LSTM, with hidden size and training-data size variations.
- **Embedding settings:** pretrained frozen, pretrained trainable and random trainable.
- **Variance:** three seeds (0–2) per final configuration, reported as mean ± std.
- **Error analysis:** mean-pool (trainable) vs. LSTM (frozen) errors, split by tweet length and fraction of unknown tokens.
- **Project 1 comparison:** `project1_comparison/` re-evaluates the Project 1 indicator + MLP system on the same split.

## Running the code

Run from the project folder. The notebook needs `torch`, `nltk`, `datasets`, `scikit-learn`, `pandas`, `numpy` and `matplotlib`.

```bash
git lfs pull                        # fetch GloVe and saved encoders (stored with Git LFS)
jupyter notebook project2.ipynb     # run the cells in order
python3 project1_comparison/compare_project1.py   # optional: Project 1 baseline
```

Saved results are included, so the analysis cells can be run without retraining.

## Results

`results/` contains per-seed scores and training histories, embedding and encoder comparisons, validation predictions, error breakdowns and learning curves. Mean over three seeds:

| System | Accuracy | Macro-F1 |
|---|---|---|
| Mean pool, pretrained frozen | 0.535 ± 0.005 | 0.412 ± 0.004 |
| Mean pool, pretrained trainable | 0.557 ± 0.006 | 0.458 ± 0.006 |
| Mean pool, random trainable | 0.543 ± 0.026 | 0.388 ± 0.032 |
| LSTM, pretrained frozen | 0.554 ± 0.009 | 0.473 ± 0.017 |
| LSTM, pretrained trainable | 0.512 ± 0.011 | 0.416 ± 0.044 |
| Project 1: indicator + MLP | 0.600 ± 0.004 | 0.510 ± 0.006 |

## Files

- `embeddings.py`, `models.py`, `training.py`: embedding loading, encoders and training loop
- `project1_comparison/`: Project 1 code and results used for the baseline
- `glove.twitter.27B.zip`: pretrained embeddings (Git LFS)

## Sources

Parts of the code are copied or adapted from [nlp-illc/nlp1-student](https://github.com/nlp-illc/nlp1-student), as indicated in the Python files.
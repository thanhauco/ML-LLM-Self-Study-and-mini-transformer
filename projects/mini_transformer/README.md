# Mini-Transformer Project

A educational project to build a small-scale transformer for character-level language modeling.

## Project Structure

- `model.py`: The core transformer architecture.
- `train.py`: Training loop and data loading logic.
- `inference.py`: Generation script to test the model.
- `data/`: Sample datasets (tiny-shakespeare).

## Learning Objectives

1. Understand tokenization (character-level).
2. Implement Causal Masking (Look-ahead mask).
3. Observe the scaling of loss during training.
4. Experiment with different `d_model` and `n_layer` configurations.

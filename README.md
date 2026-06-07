# Linear Regression from Scratch

A linear regression model built from scratch in Python — no sklearn, no shortcuts. Given a car's mileage, the model predicts its diesel cost using gradient descent.

Full write-up: [khairallah17.github.io/posts/Linear-Regression-from-Scratch](https://khairallah17.github.io/posts/Linear-Regression-from-Scratch/)

## Setup

Create and activate a virtual environment before running:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

**Train the model:**

```bash
python3 train.py
```

This reads `data.csv`, runs gradient descent, and saves the learned parameters to `params.json`.

**Run predictions:**

```bash
python3 predict.py test_data.csv
```

Loads the saved parameters and plots predictions against the actual values.

## Project Structure

```
├── train.py        # Gradient descent, normalisation, evaluation metrics
├── predict.py      # Loads params and scores new data
├── plot.py         # Visualisation helpers
├── data.csv        # Training data (mileage, diesel cost)
├── test_data.csv   # Test data
└── params.json     # Saved model parameters (generated after training)
```

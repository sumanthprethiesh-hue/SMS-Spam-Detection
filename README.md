# SMS Spam Prediction

A small Python project that trains a text classifier to label SMS messages as **spam** or **ham** (not spam).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare data

Place a CSV at `data/sms_spam.csv` with message text and labels. The loader accepts either:

- `label` and `message` columns, or
- the common UCI SMS Spam Collection format, with `v1` (label) and `v2` (message).

Labels may be `ham`/`spam` or `0`/`1`. Use a real, appropriately licensed dataset for meaningful results; the model's quality depends on its training data.

## Train

```powershell
python train.py --data data/sms_spam.csv
```

The script reports held-out accuracy, precision, recall, and F1, then saves the fitted pipeline to `models/sms_spam_model.joblib`.

## Predict

```powershell
python predict.py "Congratulations! You won a free prize."
```

You can also pass `--model path\to\model.joblib` to use a different saved model.

## Project layout

```text
train.py          Train/evaluate and save the model
predict.py        Predict a label for a message
src/sms_model.py  Data loading and model construction
data/             Put your CSV dataset here
models/           Saved model output (created during training)
```

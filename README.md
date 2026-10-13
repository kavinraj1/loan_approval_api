# Loan Approval API

A prototype that pairs a machine learning loan-approval model with GPT-4 generated FastAPI code.

## What it does

- **Model:** trains a scikit-learn `RandomForestClassifier` on the Kaggle [Credit Score Classification dataset](https://www.kaggle.com/datasets/sujithmandala/credit-score-classification-dataset). Categorical features (gender, education, marital status, home ownership) are one-hot encoded and the model is saved to `model.pkl` with joblib.
- **API generation:** an `APIGenerator` class prompts OpenAI GPT-4 to write a complete FastAPI application, cleans up the response, then saves and runs it.
- **Storage:** a `CodeStorage` class keeps every generated API in a local SQLite database (`generated_codes.db`) with timestamps.
- **Utilities:** helpers find and free the local port (5001) before the service starts.

## Tech stack

Python, FastAPI, Pydantic, OpenAI API, scikit-learn, pandas, joblib, SQLite, psutil

## Running it

1. Install the dependencies: `pip install fastapi uvicorn openai scikit-learn pandas joblib psutil python-dotenv kaggle`
2. Set your OpenAI key as an environment variable (for example in a `.env` file): `OPENAI_API_KEY=...`
3. Set up Kaggle API credentials so the notebook can download the dataset.
4. Open `app.ipynb` and run the cells in order.

import pandas as pd
import pickle

BEST_MODEL_PATH = "best_model.pkl"
FEATURE_COLUMNS = ["pclass", "sex", "age", "fare",
                   "embarked_Q", "embarked_S", "who_child"]

# load saved best model
def load_model(path=BEST_MODEL_PATH):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded")
    return model

# Predict on a DataFrame
def predict(model, input_df: pd.DataFrame):
    """
    input_df : DataFrame with columns matching FEATURE_COLUMNS
    Returns  : (predictions array, probabilities array)
    """
    preds  = model.predict(input_df)
    probas = model.predict_proba(input_df)
    return preds, probas

if __name__ == "__main__":

    model = load_model()

    # Sample passenger example
    sample = pd.DataFrame([{
        "pclass": 1,  
        "sex":1,  
        "age": 48.0,
        "fare":25.92,
        "embarked_Q": 0,
        "embarked_S": 1,   
        "who_child":  0,
    }])

    print("\nSample Input:")
    print(sample.to_string(index=False))

    preds, probas = predict(model, sample)

    label = "Survived" if preds[0] == 1 else "Not Survived"
    print(f"\nprediction  : {label}")
    print(f"probability : Survived={probas[0][1]:.2%}  |  Not Survived={probas[0][0]:.2%}")

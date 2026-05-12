from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.pkl"


def load_model(model_path: Path = MODEL_PATH):
    """
    Load the trained spam classification model from disk.

    The saved model is expected to be a complete scikit-learn Pipeline,
    containing both the text vectorizer and the classifier.

    Args:
        model_path (Path): Path to the saved model file.

    Returns:
        sklearn.pipeline.Pipeline: Trained scikit-learn pipeline used for
        spam classification.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. " "Run `python train_model.py` first."
        )

    return joblib.load(model_path)


# Load the model
model = load_model()


def predict_message(message: str) -> dict:
    """
    Predict whether a text message is spam or not spam.

    The function takes a raw SMS message, passes it through the trained
    scikit-learn pipeline, and returns both the predicted label and the
    class probabilities.

    Args:
        message (str): Raw SMS message entered by the user.

    Returns:
        dict: Dictionary containing:
            - prediction (str): Predicted class label, either "Spam" or "NotSpam".
            - not_spam_probability (float): Predicted probability for the NotSpam class.
            - spam_probability (float): Predicted probability for the Spam class.

    Raises:
        ValueError: If the input message is not a string or is empty.
    """

    if not isinstance(message, str):
        raise ValueError("Message must be a string.")

    if message.strip() == "":
        raise ValueError("Message cannot be empty.")

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    label = "Spam" if prediction == 1 else "NotSpam"
    css_class = "Spam" if prediction == 1 else "Not Spam"

    return {
        "prediction": label,
        "css_class": css_class,
        "not_spam_probability": float(probabilities[0]),
        "spam_probability": float(probabilities[1]),
    }

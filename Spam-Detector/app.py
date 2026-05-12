from flask import Flask, render_template, request
from predict import predict_message

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Render the home page and handle message classification requests.

    Returns:
        str: Rendered HTML page with the prediction result, user message,
        and any validation error.
    """

    result = None
    message = ""
    error = None

    if request.method == "POST":
        message = request.form.get("message", "")

        try:
            result = predict_message(message)
        except ValueError as exc:
            error = str(exc)

    return render_template("index.html", result=result, message=message, error=error)


if __name__ == "__main__":
    app.run(debug=True)

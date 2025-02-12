from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

def load_result():
    try:
        with open("add.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        result = load_result()

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

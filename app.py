from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Required for first deployment


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

ps = PorterStemmer()

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)


def transform_text(text):
    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    # Same preprocessing used during training
    transformed_message = transform_text(message)

    # Convert to TF-IDF
    vector_input = vectorizer.transform([transformed_message]).toarray()

    # Prediction
    prediction = model.predict(vector_input)

    if prediction[0] == 1:
        result = "Spam"
    else:
        result = "Ham"

    return render_template(
        "index.html",
        prediction_text=f"Prediction: {result}",
        entered_text=message
    )


if __name__ == "__main__":
    app.run(debug=True)
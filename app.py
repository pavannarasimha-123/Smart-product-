from flask import Flask, render_template, request
import pandas as pd
import pickle

from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load TF-IDF Vectorizer
tfidf = pickle.load(
    open("tfidf_vectorizer.pkl", "rb")
)

# Load Dataset
data = pd.read_csv(
    "train.csv",
    encoding="latin1"
)

# Unique Products
products = data[
    ["product_title"]
].drop_duplicates()

products.reset_index(
    drop=True,
    inplace=True
)

# Create Product Vectors
product_vectors = tfidf.transform(
    products["product_title"]
)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Search Route
@app.route("/search", methods=["POST"])
def search():

    query = request.form["query"]

    query_vector = tfidf.transform(
        [query]
    )

    similarity = cosine_similarity(
        query_vector,
        product_vectors
    )

    top_indices = (
        similarity[0]
        .argsort()[-10:][::-1]
    )

    results = products.iloc[
        top_indices
    ].copy()

    results["Rank"] = range(
        1,
        len(results) + 1
    )

    return render_template(
        "result.html",
        query=query,
        results=results.values.tolist()
    )

if __name__ == "__main__":
    app.run(debug=True)
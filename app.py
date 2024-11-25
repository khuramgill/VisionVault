from flask import Flask, render_template, request, redirect, url_for
from utils.preprocess import preprocess_image
from utils.pinecone_helpers import query_pinecone
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save the uploaded image
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Query Pinecone
            query_result = query_pinecone(filepath)
            return render_template("results.html", results=query_result, query_image=file.filename)

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for('static', filename=f"uploads/{filename}"))

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, redirect, url_for
import subprocess

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat")
def chat():
    # Launch Streamlit app in a separate process
    streamlit_process = subprocess.Popen(["streamlit", "run", "streamlit_app.py"])
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)

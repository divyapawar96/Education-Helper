from flask import Flask, render_template, request
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Load model and tokenizer
model_name = "t5-small"  # You can also use "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Simplify text
def simplify_text(text):
    input_text = "simplify: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Summarize text
def summarize_text(text):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=120, min_length=30, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def index():
    output_text = ""
    if request.method == "POST":
        mode = request.form.get("mode")
        input_text = request.form.get("input_text")

        if mode == "Simplify":
            output_text = simplify_text(input_text)
        elif mode == "Summarize":
            output_text = summarize_text(input_text)

    return render_template("index.html", output_text=output_text)

if __name__ == "__main__":
    app.run(debug=True)

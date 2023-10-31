import os
import fitz  # PyMuPDF
import nltk
import torch
from nltk.tokenize import sent_tokenize
from transformers import BartTokenizer, BartForConditionalGeneration
from flask import Flask, render_template, request

# Download the 'punkt' package from NLTK
nltk.download('punkt')

app = Flask(__name__)

UPLOAD_FOLDER = 'pdfs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def preprocess_text(text):
    
    sentences = sent_tokenize(text)
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        if sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)
    return ' '.join(unique_sentences)

def summarize_text(text, context, max_length, max_words):
    if context is not None:
        
        input_text = preprocess_text(context + " " + text)
    else:
        
        input_text = preprocess_text(text)

    
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=max_length, truncation=True)
    summary_ids = model.generate(inputs, num_beams=4, max_length=max_length, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    
    num_words = len(summary.split())


    if num_words > max_words:
        sentences = sent_tokenize(summary)
        truncated_summary = ""
        for sentence in sentences:
            if len(truncated_summary) + len(sentence) <= max_words:
                truncated_summary += sentence + " "
            else:
                break
        summary = truncated_summary.strip()

    return summary.strip()

def extract_text_from_pdf(file_path):
    try:
        
        pdf_file = fitz.open(file_path)

       
        pdf_text = ""
        for page_num in range(pdf_file.page_count):
            page = pdf_file.load_page(page_num)
            pdf_text += page.get_text()

        pdf_file.close()

        
        pdf_text = pdf_text.encode('ascii', 'ignore').decode()

        return pdf_text
    except Exception as e:
        return f"Error occurred while processing the PDF: {str(e)}"

def summarize_pdf(file_path, context, max_words=500):
    try:
      
        pdf_text = extract_text_from_pdf(file_path)

        sentences = sent_tokenize(pdf_text)
        chunks = []
        chunk = ""
        for sentence in sentences:
            if len(chunk) + len(sentence) <= max_words:
                chunk += sentence + " "
            else:
                chunks.append(chunk.strip())
                chunk = sentence + " "
        if chunk:
            chunks.append(chunk.strip())

        summaries = []
        for chunk in chunks:
            summarized_chunk = summarize_text(chunk, context, max_length=512, max_words=max_words - len(" ".join(summaries).split()))
            summaries.append(summarized_chunk)

            if len(" ".join(summaries).split()) >= max_words:
                break

        summarized_text = " ".join(summaries)

        return summarized_text
    except Exception as e:
        return f"Error occurred while processing the PDF: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['pdf_file']
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            context = request.form.get('context')
            summarized_text = summarize_pdf(file_path, context, max_words=500)

            return render_template('index.html', summary=summarized_text)

    return render_template('index.html', summary=None)

if __name__ == '__main__':
    app.run(debug=True)

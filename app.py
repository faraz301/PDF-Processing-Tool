from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import fitz  
import numpy as np
from PIL import Image
import tempfile
import csv
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from transformers import pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_path):
    doc = fitz.opSen(pdf_path)
    sections = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        sections.append((page_num + 1, text))
    return sections

def pdf_page_to_image(page):
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def detect_rois_in_pdf(pdf_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    doc = fitz.open(pdf_path)
    rois = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        img = pdf_page_to_image(page)
        img_np = np.array(img)
        outputs = predictor(img_np)
        rois.append(outputs["instances"])
    return rois

def process_pdf(pdf_file):
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf_file.save(temp_pdf.name)
    pdf_path = temp_pdf.name

    sections = extract_text_from_pdf(pdf_path)
    
    csv_data = [['Page Number', 'Text']]
    for page_num, text in sections:
        csv_data.append([page_num, text.strip()])

    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=app.config['UPLOAD_FOLDER'])
    with open(temp_csv.name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

    combined_text = " ".join([text for _, text in sections])

    return temp_csv.name, combined_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            return redirect(request.url)
        pdf_file = request.files['pdf_file']
        if pdf_file.filename == '':
            return redirect(request.url)
        if pdf_file:
            csv_file, combined_text = process_pdf(pdf_file)
            return redirect(url_for('result', filename=os.path.basename(csv_file), combined_text=combined_text))
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(file_path, 'r') as csvfile:
        csv_data = list(csv.reader(csvfile))
    combined_text = request.args.get('combined_text')
    summary = ""
    if combined_text:
        max_length = 3000
        truncated_text = combined_text[:max_length]
        summary = summarizer(truncated_text, max_length=500, min_length=400, do_sample=False)[0]['summary_text']
    return render_template('result.html', csv_data=csv_data, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)

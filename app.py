from flask import Flask,render_template,request, send_file
from transformers import pipeline

import requests
import pdfplumber
import os
import easyocr
import cv2
import numpy as np
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from googletrans import Translator



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('base.html')


@app.route('/ocr', methods = ['POST','GET'])
def captureText():
    if request.method == 'POST':
        input_file = request.files['imageFile']

        #if file uploaded is an image
        if input_file.content_type in ['image/jpg','image/png','image/jpeg']:

            if(input_file.filename):
                imagePath = "./images/" + input_file.filename
            else:
                return render_template('base.html')

            lang = request.form['drop-down']
            input_file.save(imagePath)
            def cleanup_text(text):
                # strip out non-ASCII text, we can draw the text on the image
                return "".join([c if ord(c) < 128 else "" for c in text]).strip()

            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.GaussianBlur(image,(5,5),0)
            kernel3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel3)
            reader = easyocr.Reader(['en', lang])
            results = reader.readtext(image)

            main_text = ''
            for (bbox, text, prob) in results:
                # display the OCR'd text and associated probability
                main_text = main_text + '\t' + (text)
                print("[INFO] {:.4f}: {}".format(prob, text))

                # unpack the bounding box
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))
                text = cleanup_text(text)
                cv2.rectangle(image, tl, br, (0, 255, 0), 2)

            print(main_text)
            os.remove(imagePath)
            return render_template('base.html', output = main_text)

        #if file uploaded is a pdf
        elif input_file.content_type == 'application/pdf':
            if(input_file.filename):
                filePath = "./pdf/" + input_file.filename
            else:
                return render_template('base.html')

            pdf_text = ''
            input_file.save(filePath)
            with pdfplumber.open(filePath) as pdf:
                for page in pdf.pages:
                    pdf_text = pdf_text + page.extract_text()
            textFile = open('./textfiles/'+ 'pdf2Text.txt' , 'w' ,encoding='utf-8')
            textFile.write(pdf_text)
            textFile.close()
            os.remove(filePath)
            onSuccess = 'download'
            
            return render_template('base.html', download = onSuccess)

        else:
            return render_template('base.html')
    # else:
    #     return render_template('base.html')
    
    #return render_template('base.html')


@app.route('/ocr/translate', methods=['POST','GET'])
def translateText():
    translator = Translator()
    if request.method == 'POST':
        translation = ''
        text = request.form['input-text']
        lang_tr = request.form['drop-down-tr']
        if text:
            translation = translator.translate(text, dest=lang_tr)

            print(translation)
            return render_template('base.html', translation = translation.text)
        else:
            def_text = request.form['output-text']
            return render_template('base.html', translation = def_text)
        
            
    else:
        return render_template('base.html')


@app.route('/ocr/download', methods = ['GET','POST'])
def download():
    path = 'textfiles/pdf2Text.txt'
    if path:
        return send_file(path, as_attachment=True)
    else:
        return f'wrong'

@app.route('/ocr/summary/downloadS', methods=['GET','POST'])
def downloadSummary():
    path = 'textfiles/textSummary.txt'
    if path:
        return send_file(path, as_attachment=True)
    else:
        return f'error'


@app.route('/ocr/summary', methods = ['GET','POST'])
def summary(): 
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    with open('textfiles/pdf2Text.txt','r',encoding='utf-8') as f:
        text = f.read()
        print(text)
        str = text.split()
        words = 0
        for i in str:
            words = words + 1


    # tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
    # model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
    # with open('textfiles/pdf2Text.txt','r',encoding='utf-8') as f:
    #     text = f.read()
    # tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    # summary = model.generate(**tokens)
    # main_text = tokenizer.decode(summary[0])
    # with open('textfiles/textSummary.txt','w',encoding='utf-8') as file:
    #     file.write(main_text)


    main_text = (summarizer(text, max_length=round(words*60/100), min_length=round(words*20/100), do_sample=False))
    for i in main_text:
        text_dict = i
    text = ''
    with open('textfiles/textSummary.txt','w',encoding='utf-8') as file:
        file.write(text_dict['summary_text'])
    onSuccess = 'download summary'
    return render_template('base.html', summary = onSuccess)


if __name__ == '__main__':
    app.run(debug=True)

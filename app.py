from turtle import down
from flask import Flask,render_template,request

import pdfplumber
import os
import easyocr
import cv2
import numpy as np

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

            text = ''
            input_file.save(filePath)
            with pdfplumber.open(filePath) as pdf:
                for page in pdf.pages:
                    text = text + page.extract_text()
            textFile = open('./textfiles/'+ 'pdf2Text.txt' , 'w' ,encoding='utf-8')
            textFile.write(text)
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
        if text:
            translation = translator.translate(text)

            print(translation)
            return render_template('base.html', translation = translation.text)
        else:
            def_text = request.form['output-text']
            return render_template('base.html', translation = def_text)
        
            
    else:
        return render_template('base.html')



if __name__ == '__main__':
    app.run(debug=True)

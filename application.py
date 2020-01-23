from flask import Flask, render_template, request
from face_detection import face_detect
app = Flask(__name__)

@app.route('/')
def student():
    return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        img = face_detect(result['url'])
        img.save("templates/img.jpg")
        return render_template("result.html", result=result)

if __name__ == '__main__':
    app.run(debug = True)

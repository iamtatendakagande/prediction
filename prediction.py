from flask import Flask, render_template, redirect, url_for, request, session, send_file, flash
import pandas as pd
from machine.source.userInputModel import predict

app = Flask(__name__, template_folder='./public')
app.config['UPLOAD_FOLDER'] =  'static/download'

@app.route('/', methods=['GET'])
def index():
    return render_template('/index.html')

@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    if (request.method == "POST"):
        file = request.files['file']
        if file.filename != '':
            try:
                input = pd.read_csv(file)
                print(input.iloc[0, 0])#index 1
                print(input.iloc[0, 1])#question 1
                output = predict.userInput(input)
                return render_template('prediction/output.html', price = output)
            except Exception as e:
                print("An error occurred:", e)
                return render_template('prediction/prediction.html')
        else:
            print("Please attach soemething")
            return render_template('prediction/prediction.html') 
    else:
        return render_template('prediction/prediction.html')
       
@app.route("/download")
def download():
    return send_file(
        "./static/downloads/prediction-template.zip",
        mimetype="application/zip",
        download_name="prediction-template.zip",
        as_attachment=True,) 
              
if __name__ == '__main__':
    app.run(debug=True)
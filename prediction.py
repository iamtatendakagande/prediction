from flask import Flask, render_template, redirect, url_for, request, session, send_file, flash
import pandas as pd
import os
from machine.source.userInputModel import predict

app = Flask(__name__, template_folder='./public')
app.config['UPLOAD_FOLDER'] =  'static/download'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24)

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
                print(input.iloc[0, 0])
                print(input.iloc[0, 1])
                output = predict.userInput(input)
                flash(f'✅ The expected rent is: ${output:.2f}', 'success') 
                return render_template('prediction/prediction.html')
            
            except Exception as e:
                flash(f"❌ An error occurred while processing the file: {e}", 'error')
                return render_template('prediction/prediction.html')
        else:
            flash("⚠️ Please attach a file before uploading.", 'warning')
            return render_template('prediction/prediction.html') 
    else:
        return render_template('prediction/prediction.html')
       
@app.route("/download")
def download():
    return send_file(
        "./static/downloads/model-input-data.zip",
        mimetype="application/zip",
        download_name="model-input-data-template.zip",
        as_attachment=True,) 
              
if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template,request
from keras.preprocessing import sequence
import keras 
import tensorflow as tf
import os
import numpy as np
from use_model import shakes_ai

ai = shakes_ai()
inp = 'I am Romeo and '
app = Flask(__name__)

@app.route("/")
def hello():
  return render_template('home.html')

@app.route('/',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      prompt = request.form.get("prompt")
      print(prompt)
      return render_template("home.html",prompt=prompt,content = ai.generate_text(prompt))

if __name__ == "__main__":
  app.run()
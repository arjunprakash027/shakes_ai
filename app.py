from flask import Flask
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
  return (ai.generate_text(inp,1000))

if __name__ == "__main__":
  app.run()
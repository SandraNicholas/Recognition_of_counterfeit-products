import re
from datetime import datetime
import pickle

from flask import Flask,jsonify,request
from flask import render_template


import numpy as np
from numpy import asarray
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.utils import load_img
from keras.models import load_model
import cv2
import PIL
import os
from keras.preprocessing import image
from sentence_transformers import SentenceTransformer
#from sklearn import preprocessing
from sklearn.decomposition import PCA
from PIL import Image
import pandas as pd

app = Flask(__name__)

autoencoder=load_model('model.h5')
text_model=load_model('meta_model_fine.h5')
model_classifier=load_model('model_classifier.h5')

with open('lefinal.pkl', 'rb') as f:
    le = pickle.load(f)
with open('pcafinal.pkl', 'rb') as f:
    pca = pickle.load(f)
target_img = os.path.join(os.getcwd() , 'static/images')


@app.route("/")
def index(name = None):
    return render_template(
        "index.html",
    )

def preprocess_image(image):
    img = PIL.Image.open(image)
    img = img.resize((128, 128))
    gray_img = img.convert("L")
    data = asarray(gray_img)
    return data

def compute_mse(image, recon):
    mse = np.mean((image - recon) ** 2)
    return mse
def resize_img(name, resize_config = 150):
    #temp_path = f"/Users/kenny/Dropbox/GA/capstone_assets/images/det/{is_rep}_det/{name}"   
    im = Image.open(name)                                      
    
    #calculate aspect ratio
    org_size = im.size                                              
    ratio = float(resize_config)/max(org_size)                      
    new_size = tuple([int(x * ratio) for x in org_size])            

    new_im = im.convert(mode='RGB')                                 
    new_im = new_im.resize(new_size, Image.LANCZOS)                 

    canvas = Image.new("RGB", (resize_config, resize_config))       
    canvas.paste(new_im, ((resize_config-new_size[0])//2,           
                         (resize_config-new_size[1])//2))
    img_arr_det=np.array(canvas)
    img_arr_d2 = img_arr_det.astype('float32')
    img_arr_d2/=255
    img_arr_d2 = np.expand_dims(img_arr_d2, axis=0)
    #return img as array
    return img_arr_d2 



# Route for predicting with the autoencoder
@app.route("/predict", methods=["POST"])
def predict():
    # Check if an image was uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Read the image file
    image_file = request.files["image"]

    # Preprocess the image
    image = resize_img(image_file)
    #image = image / 255
    #image = np.expand_dims(image, axis=0)
    answer=model_classifier.predict(image)
    answer=round(answer[0][0])

    # Predict using the autoencoder model
    #decoded = autoencoder.predict(image)
    #result = decoded[:, :, :, 0]

    # Compute the mean squared error
    #mse = compute_mse(image, result)

    # Return the result as a JSON response
    return jsonify({"mse": answer})



st_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the label encoder
#le = preprocessing.LabelEncoder()

@app.route('/predict_text', methods=['POST'])
def predict_text():
    # Get the request data
    data = request.get_json()
    

    # Preprocess the data
    cleaned_title = data['title']
    price = data['price']
    rating = data['rating']
    brand = data['brand']
    # preprocess the title
    
    # encode the title
    title_vector = st_model.encode(cleaned_title)
    print('hello')
    title_vector = pca.transform(title_vector)
    # encode the brand using label encoder
    brand_encoded = le.transform([brand])[0]
    #print(brand_encoded)
    
    # create the input array
    input_arr = pd.DataFrame([price, rating, title_vector, brand_encoded])
    #input_arr = np.asarray(input_arr).astype(np.float32)
    
    # make the prediction
    prediction = text_model.predict(input_arr)
    
    # return the prediction as a JSON object
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
''' 


    if request.method == 'POST':
        file = request.files['file']
        print ('Hellooo')
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            
            file.save(file_path)
            #img = read_image(file_path) #preproressing method
            #class_prediction=model.predict(img) 
            #print('hello')
            
            #classes_x=np.argmax(class_prediction,axis=1)
            #if classes_x == 0:
            #  fruit = "Apple"
            #elif classes_x == 1:
            #  fruit = "Banana"
            #else:
            #  fruit = "Orange"
            #'fruit' , 'prob' . 'user_image' these names we have seen in predict.html.
            #return render_template('predict.html', fruit = fruit,prob=class_prediction, user_image = file_path)
        #else:
            #return "Unable to read the file. Please check file extension"
            return "File is read successfully"
@app.route('/my-link/')
def my_link():
  print ('I got clicked!')

  return 'Click.'
@app.route('/api')
def api():
    print('hello')
    data = {'message': 'Hello, world!'}
    return jsonify(data)
def upload():
    file=request.files['sneakerImage']
    return 'Image uploaded successfully' '''

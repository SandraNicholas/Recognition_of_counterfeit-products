# Group8_Recognition_of_counterfeit-products
Project Name: "Recognition of counterfeit products"

Project Overview: In this project, we aim to distinguish the original products and counterfeit products from images of products and metadata of the product.

Dataset: We have used to datasets for this project. The first one is a collection of images that contain both original images and counterfeit images.  We extracted the images from Reddit. The second dataset contains the metadata about the product. The dataset contains information such as Title, Brand, rating, and review count.

Tools and Technologies Used: For this project, we have used Python programming language and various libraries such as Pandas, NumPy, matplotlib, and Scikit-learn. We have also used Jupyter Notebook for coding and analysis.

Project Structure: The project consists of the following files:
1. ”Reddit_scrape.ipynb” – This file contains the Python code for scraping the counterfeit images. 

2. “IBF_Data_Scrapping.ipynb” – This file contains Python code to web scrape the original images.

3. “counterfiet_data_scrape.ipynb” – This file contains the Python code for web scraping the metadata.

4. "metadata.csv" - This file contains the information scraped from different websites about the products used in the project.

5. "lefinal.pkl" - This file contains the Python code for the label encoding.

6. "meta_mode_final.h5" - This file contains the Python code for metadata classification using an Artificial neural network.

7. “model.h5” – is the Python code for Autoencoder and decoder for image classification.

8. “model_classifier.h5” – It is the Python code for Image classification using CNN.

9. “pcafinal.pkl” – It is the Python code for Principle Component Analysis used for Dimensionality reduction.


Instructions for Running the Project: To run the project, please follow the below steps:

1. Clone the project repository to your local machine.

2. Run app.py in the virtual environment  .venv.

3. Output Local IP address has to be copied and displayed.

4. Upload the image from the local system.

5. It will display whether the product is original or counterfeit.

Conclusion: Convolutional Neural Networks (CNNs) have shown great potential in recognizing counterfeit products by analyzing their visual features. Through the process of training CNN on a large dataset of genuine and counterfeit products, the network can learn to distinguish between the two and accurately classify new products as either genuine or counterfeit. Since some products, they use the same image for both counterfeit and original products, it may be difficult to predict output only on the basis of images. So, we did metadata classification and combined both models.




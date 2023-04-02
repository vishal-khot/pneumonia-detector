# PNEUMONIA DETECTION USING CHEST X-RAY
Pneumonia is a common lung infection in not only our country but also the world. Anyone can develop pneumonia. The people most at risk of developing pneumonia are young  children, older adults, and people with preexisting medical conditions that weaken the immune system. These same groups of people are also at increased risk of developing complications of pneumonia. To diagnose pneumonia, a doctor will usually ask about a person’s symptoms and medical history and carry out a physical examination. The physical exam may include listening to the chest through a stethoscope and measuring blood oxygen levels using a pulse oximeter attached to the finger. However, a more accurate way to detect the disease would be using chest x ray images. A radiologist is needed to interpret the chest x-ray and determine if a patient has pneumonia. We have developed a deep learning model that detects pneumonia in front chest x ray images. Our model outperforms existing methods in terms of both efficiency and accuracy.

## PROPOSED METHODOLOGY

### Approach 1
We have built and trained our own neural network with the architecture depicted
in the following image.
<p align="center">
     <img src="https://user-images.githubusercontent.com/55139240/187025886-edc17aed-2626-4278-ae18-9088367a7a9f.png">
</p>
The neural network was trained for 50 epochs with model checkpoints.
Only the best weights encountered while training were saved.
436 images were used for testing.<br><br>
<b>Test accuracy</b> is 91.5<br>
<b>Validation Accuracy</b> is 90.4<br>
<b>Train accuracy</b> is 99.69<br>

### Approach 2
The model architecture is the same as in approach 1. All the images were applied a transformation, to enhance the contrast of the images. A Previous approach had indicated better results with better contrast in the chest x-ray images. The transformation applied is the convertScaleAbs in opencv. For example, the input and the transformed output is shown in the following image. Note the enhanced contrast in the result.
<p align="center">
     <img src="https://user-images.githubusercontent.com/55139240/187026080-ecf82cbc-33ed-47db-a6b3-3fe878f72b58.png">
</p>

### Approach 3
Progressive resizing is a well known method to enhance the results obtained from an image classification model. The weights of the dense layers are used as is, and the dense layers are frozen while training. The image size used while training is 128x128. The model was trained for 200 epochs in this way.<br>
**Progressive Image Resizing** is the technique to sequentially resize all the images while training the CNN models on smaller to bigger image sizes.A great way to use
this technique is to train a model with smaller image size say 64 x 64, then use the weights of this model to train another model on images of size 128 x 128 and so
on. Each larger-scale model incorporates the previous smaller-scale model layers and weights in its architecture, and thereby results in fine-tuning the final model
as well as increasing the accuracy score.

## TOOLS AND TECHNOLOGIES USED

• **Jupyter Notebook** - The Jupyter Notebook is the original web application for
creating and sharing computational documents. It offers a simple, streamlined,
document-centric experience. Python code can be run interactively on Jupyter
notebooks.<br>
• **Python** - Python is a high-level, interpreted, general-purpose programming
language. Its design philosophy emphasizes code readability with the use of
significant indentation.<br>
• **OpenCV** - OpenCV is a library of programming functions mainly aimed at realtime
computer vision. Originally developed by Intel, it was later supported
by Willow Garage then Itseez. The library is cross-platform and free for use
under the open-source Apache 2 License.<br>
• **Numpy** - NumPy is a library for the Python programming language, adding
support for large, multi-dimensional arrays and matrices, along with a large
collection of high-level mathematical functions to operate on these arrays.<br>
• **Tensorfow** - TensorFlow is a free and open-source software library for machine
learning and artificial intelligence. It can be used across a range of tasks but
has a particular focus on training and inference of deep neural networks.<br>
• **Flask** - Flask is a micro web framework written in Python. It
is classified as a microframework because it does not require particular tools
or libraries. It has no database abstraction layer, form validation, or any
other components where pre-existing third-party libraries provide common
functions.<br>

## RESULTS

All the models were tested on the same testing dataset with 530 images. The following figure shows the results obtained by testing the model from the three approaches.
<p align="center">
     <img src="https://user-images.githubusercontent.com/55139240/187026337-3234f522-b08b-41df-8cc5-1ff649be96a8.png">
</p>
Approach 3 gives the best result among the three approaches with the highest accuracy, precision and recall.<br><br>

<p align="center">
     <img src="https://user-images.githubusercontent.com/55139240/187026393-ec09647b-0c59-44f1-af9e-3eb99a342c2f.png" width="800" height="400"><br>
     Variation of accuracy and loss with number of epochs in approach 2<br>
</p>

<p align="center">
     <img src="https://user-images.githubusercontent.com/55139240/187026501-189f9e03-9353-4add-a6ef-769eda28f8f4.png" width="800" height="400"><br>
     Variation of accuracy and loss with number of epochs in approach 3<br>
</p>
<br>
As the size of our model is just 3.09 MB (transfer learning based models are hundreds of MB in size), our model can be used with a mobile app even on low end mobile phones.

## RUNNING THE CODE
Clone the repository or download the zip file <br>
```git clone https://github.com/VallishaM/chest-xray.git```
<br><br>
Go into the backend directory<br>
```cd backend```
<br><br>
Run the python file named app.py<br>
```python3 app.py```
<br><br>
Click on the link in the terminal to open the web app in the browser.

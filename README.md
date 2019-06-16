# Grab_Vision_Challenge
I. Introduction
This is my framework: <br/>
![Alt text](pipline.png?raw=true "The pipeline of my approach")

II. Testing
1. Requirements
   - Ubuntu 16.04
   - python 2.7 
   - git
   - GPU: memory >= 10GB
   - (optional) anaconda3 then create the python 2.7:( tested on my machine)
      + conda create -n Grab python=2.7 anaconda
      + conda activate Grab
2. Clone this project: git clone https://github.com/phunghx/Grab_Vision_Challenge.git
3. Build library:
   - Edit the file Makefile in darknet folder based on your system. This library was tested on GPU with cuda 9.0, cudnn 7.3
   - Build darknet: bash build_darknet.sh
4. Install library for python:
   - pip install -r requirements.txt
5. Get  pre-trained models:
   -  bash get_model.sh
   - (optional - if the upper command isn't working. Please download two models with the links in file get_model.sh and put on appropriate directories   

6. Predict for all images in a folder
   - cd fast-MPN-COV/
   - bash run_test.sh &lt;input of images folder&gt; &lt;output csv file&gt;
   <br/>
   Examples: bash run_test.sh ../test_imgs ../submission/final_prediction.csv
   <br/>
   
   - The ouput csv file has the header with
      + The first column is filename, which is appropriate with each filename in images folder
      + The second column is the prediction class
      + From the third column to end is the confidence score for each class
      + Headers are: filename,prediction,AM General Hummer SUV 2000,Acura RL Sedan 2012, ...
      + Please look up the detail of output file in submission/final_prediction.csv
   

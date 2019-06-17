# Grab_Vision_Challenge
I. Introduction <br/>
This is my framework: <br/>
<br/>
![Alt text](pipline.png?raw=true "The pipeline of my approach")
<br/>
My solution has two main steps. First, the detector, namely yolov3, will detect the car from an image. Then the classification based on Fast MPN-COV is used to recognize the details of the vehicles from images, including make and model. I assume that there is only one car in an image. This approach created a unified of using both detector and classifier to increase the performance.

II. Testing
<br/> My framework can handle with JPEG and PNG image files <br/>
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
   - Edit the file Makefile in darknet folder based on your system. This library was tested on GPU with cuda 9.0, cudnn 7.3. If you have a GPU, please set the value GPU=1 and CUDNN=1 in Makefile
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

III. References
1. https://github.com/pjreddie/darknet
2. https://github.com/jiangtaoxie/fast-MPN-COV

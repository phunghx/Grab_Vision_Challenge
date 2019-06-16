mkdir -p data/data_detection
mkdir -p data/cars/train
mkdir -p data/cars/val
mkdir -p models/model_detection
mkdir -p data/Grab_Challenge
wget --no-check-certificate https://googledrive.com/host/1N-E3tRJxxfSZcXmn82RiuMu65MulD6hP/yolov3_car_10000.weights -O models/model_detection/yolov3_car_10000.weights
wget --no-check-certificate https://googledrive.com/host/1N-E3tRJxxfSZcXmn82RiuMu65MulD6hP/model_best.pth -O models/model_best.pth

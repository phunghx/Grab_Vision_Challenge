mkdir -p data/data_detection
mkdir -p data/cars/train
mkdir -p data/cars/val
mkdir -p models/model_detection
mkdir -p data/Grab_Challenge
wget --no-check-certificate https://drive.google.com/file/d/10OpJGEVOJLSxLHr7rPHC1cSyKGa9CZC1/view?usp=sharing -O models/model_detection/yolov3_car_10000.weights
wget --no-check-certificate https://drive.google.com/file/d/1SYuy5dfU5F8oNqVkqsiCbqjZwHnoaiJP/view?usp=sharing -O models/model_best.pth

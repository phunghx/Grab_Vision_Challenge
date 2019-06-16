mkdir -p data/data_detection
mkdir -p data/cars/train
mkdir -p data/cars/val
mkdir -p models/model_detection
mkdir -p data/Grab_Challenge
gdrivedl https://drive.google.com/file/d/1SYuy5dfU5F8oNqVkqsiCbqjZwHnoaiJP/view?usp=sharing models/model_best.pth.tar
gdrivedl https://drive.google.com/file/d/10OpJGEVOJLSxLHr7rPHC1cSyKGa9CZC1/view?usp=sharing models/model_detection/yolov3_car_10000.weights

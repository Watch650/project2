from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained mode

model.train(data=r'C:\Users\Asus\Desktop\scul\B2\Machine Learning and Data Mining\project2\data\type_dataset',   # load the dataset
            epochs=20, imgsz=128)

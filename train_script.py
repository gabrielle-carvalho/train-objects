from ultralytics import YOLO
import torch

def train_yolov8(model_name, data_yaml, epochs, batch_size, lr, optimizer):
    torch.cuda.empty_cache()
    model = YOLO(model_name)   
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        lr0=lr,
        optimizer=optimizer,
        device='cuda'  
    )
    
model_name = 'yolov8m.pt'
data_yaml = '/home/bill7/train_objects/data.yaml'
epochs = 200 # 119 epochs completed in 3.431 hours.
batch_size = 4
lr = 0.001
optimizer = 'SGD'

train_yolov8(model_name, data_yaml, epochs, batch_size, lr, optimizer)


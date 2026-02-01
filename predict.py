import torch
import cv2
from torchvision import transforms
from torch.autograd import Variable

def predict_image(model, image_path, transform):
    model.eval()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = transform(img)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

        print(f'Predicted class: {predicted.item()}')

if __name__ == "__main__":
    # Đường dẫn tới ảnh dự đoán
    image_path = r"E:\hocphan\neuron\nhandanglogo\Nhan dien Logo xe\preprocessed_data\Ferrari\1.png"

    # Đường dẫn tới mô hình đã được lưu
    model_path = "vehicle_model.pth"

    # Áp dụng các biến đổi cho ảnh dự đoán
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Khởi tạo mô hình và tải trọng số đã được lưu
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))

    # Dự đoán ảnh
    predict_image(model, image_path, transform)

import cv2
import os

def preprocess_images(input_folder, output_folder):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        output_class_path = os.path.join(output_folder, class_folder)

        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            output_path = os.path.join(output_class_path, image_file)

            # Đọc ảnh và chuyển đổi thành ảnh xám
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Áp dụng bộ lọc Canny để tách biên
            edges = cv2.Canny(img, 50, 150)

            # Lưu ảnh đã tiền xử lý
            cv2.imwrite(output_path, edges)

if __name__ == "__main__":
    input_folder = r"E:\hocphan\neuron\nhandanglogo\Nhan dien Logo xe\preprocessed_data"  # Thay đổi đường dẫn tới thư mục chứa ảnh của bạn
    output_folder = "preprocessed_data"
    
    preprocess_images(input_folder, output_folder)
    print("Tiền xử lý ảnh thành công!")

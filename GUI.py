from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
# class_names = ['Dang cho nhan dang ...','Vinfast','Ferrari','Toyota']
class_names = [line.strip() for line in open("labels.txt")]

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)

def predict_image(image):
    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)
    normalized_image = (image_array / 127.5) - 1

    prediction = model.predict(normalized_image,verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def update_display():
    ret, frame = camera.read()

    if ret:
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        video_canvas.create_image(0, 0, anchor='nw', image=photo)
        video_canvas.photo = photo

        prediction, confidence = predict_image(frame)
        # result_label.config(text=f"Du Doan : {prediction[2:]} — Do chinh xac: {confidence:.1f}%")
        # result_label.config(text=f"Du Doan : {prediction} — Do chinh xac: {confidence * 100:.1f}%")
        if prediction == class_names[0]:
            text = "Đang chờ nhận dạng..."
        else:
            text = f"Dự đoán: {prediction} — Độ chính xác: {confidence * 100:.1f}%"
        result_label.config(text=text)

    root.after(10, update_display)

root = tk.Tk()
root.title("ỨNG DỤNG NHẬN DẠNG LOGO CÁC HÃNG XE Ô TÔ")

main_frame = tk.Frame(root)
main_frame.pack(padx=10, pady=10)

video_canvas = tk.Canvas(main_frame, width=640, height=480)
video_canvas.pack()

result_label = tk.Label(main_frame, text="Class: ", font=("Helvetica", 16))
result_label.pack()


update_display()

root.mainloop()

camera.release()
cv2.destroyAllWindows()

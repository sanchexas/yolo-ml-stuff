from ultralytics import YOLO
from tkinter import *
import cv2 
from PIL import Image, ImageTk 

# Доступ к камере
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
# Сетим ширину и высоту получаемого изображения с камеры
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 700) 
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 900) 
  
app = Tk()
# Инициализация лейбла
label_widget = Label(app)
label_widget.pack() 

# Получает на вход изображение .jpg, определяет на нем элементы, возвращает координаты элементов
def get_cords_from_yolo(unprepared_image):
    # Уже заранее обученная модель
    model = YOLO("../yolov8m.pt")
    # Сетим изображение для определения координат и в принципе определения того, что изображено на картинке
    results = model.predict(unprepared_image)

    finalResult = results[0]

    box = finalResult.boxes[0]

    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    class_id = box.cls[0].item()
    conf = box.conf[0].item()
    print("-----Тип объекта:", class_id)
    print("-----Координаты:", cords)
    print("-----Вероятность:", conf)
    return cords

def run_camera():
    # Чтение изображения с камеры при помощи CV2 (tkinter не может получить доступ к камере, только через CV2)
    _, frame = vid.read()
    # Преобразование в цветное изображение
    colorful_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Получение координат объекта на фото через модель YOLO
    prepared_cords_array = get_cords_from_yolo(colorful_image)
    # Начальные координаты = (лево, верх)
    start_cords = [prepared_cords_array[0], prepared_cords_array[1]]
    # Конечные координаты = (право, низ)
    end_cords = [prepared_cords_array[2], prepared_cords_array[3]]
    # Сетим изображение, начальные координаты, конечные координаты, цвет рамки и ширину рамки, для отображения рамки
    cv2.rectangle(colorful_image, start_cords, end_cords, (255, 0, 0), 2)
    # Преобразование массива буфера в файл изображения
    captured_image = Image.fromarray(colorful_image) 
    # Отображение преобразованного в файл изображения в лейбл tkinter
    photo_image = ImageTk.PhotoImage(image=captured_image) 
    label_widget.photo_image = photo_image 
    label_widget.configure(image=photo_image) 
    label_widget.after(2000, run_camera) 

# Создание кнопки и привязка действия запуска функции run_camera()
button1 = Button(app, text="run camera", command=run_camera) 
button1.pack() 

# Запуск приложения tkinter - Обязательная функция
app.mainloop()
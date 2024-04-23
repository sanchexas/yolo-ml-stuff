from ultralytics import YOLO
from tkinter import *
import cv2 
import random
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

# Рандомный RGB
def get_random_rgb():
    return (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))

# Получает на вход изображение .jpg, определяет на нем элементы, возвращает координаты элементов
def get_all_elements_from_yolo(unprepared_image):
    detected_elements_array = []
    # Уже заранее обученная модель
    model = YOLO("./runs/detect/train7/weights/best.pt")
    # Сетим изображение для определения координат и в принципе определения того, что изображено на картинке
    results = model.predict(unprepared_image)

    finalResult = results[0]
    print('-------------------ОБНАРУЖЕННЫЕ ОБЪЕКТЫ НИЖЕ-------------------')
    for box in finalResult.boxes:
        class_id = finalResult.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        # Пушим в массив результата
        detected_elements_array.append({'name': class_id, 'cords': cords, 'conf': conf})
        print("|-ТИП ОБЪЕКТА:", class_id)
        print("|-КООРДИНАТЫ:", cords)
        print("|-ВЕРОЯТНОСТЬ:", conf)
        print("---")
    return detected_elements_array

def run_camera():
    # Чтение изображения с камеры при помощи CV2 (tkinter не может получить доступ к камере, только через CV2)
    _, frame = vid.read()
    # Преобразование в цветное изображение
    colorful_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_elements_array = get_all_elements_from_yolo(colorful_image)
    # Обводим и даем название каждому элементу, добавляем рамку и текст на фото
    i = 0
    for element in detected_elements_array:
        # Начальные координаты = (лево, верх)
        start_cords = [element['cords'][0], element['cords'][1]]
        # Конечные координаты = (право, низ)
        end_cords = [element['cords'][2], element['cords'][3]]
        name = element['name']
        if name == 'cell phone': name = 'телефон'
        elif name == 'person': name = 'человек'
        elif name == 'bottle': name = 'бутылка'
        elif name == 'egg': name = 'яйцо'
        # Сетим изображение, начальные координаты, конечные координаты, цвет рамки и ширину рамки, для отображения рамки
        # rgb = get_random_rgb()
        i += 1
        cv2.rectangle(colorful_image, start_cords, end_cords, (255,0,0), 2)
        cv2.putText(colorful_image, f'{i} {name} {element['conf']}', (start_cords[0], start_cords[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    # Преобразование массива буфера в файл изображения
    captured_image = Image.fromarray(colorful_image) 
    # Отображение преобразованного в файл изображения в лейбл tkinter
    photo_image = ImageTk.PhotoImage(image=captured_image) 
    label_widget.photo_image = photo_image 
    label_widget.configure(image=photo_image) 
    label_widget.after(100, run_camera) 

# Создание кнопки и привязка действия запуска функции run_camera()
button1 = Button(app, text="run camera", command=run_camera) 
button1.pack() 

# Запуск приложения tkinter - Обязательная функция
app.mainloop()
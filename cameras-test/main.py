import cv2   
from ultralytics import YOLO
import threading

# define a video capture object 
def get_all_elements_from_yolo(unprepared_image):
    detected_elements_array = []
    # Уже заранее обученная модель
    model = YOLO("best.pt")
    # Сетим изображение для определения координат и в принципе определения того, что изображено на картинке
    results = model.predict(unprepared_image)

    finalResult = results[0]
    # print('-------------------ОБНАРУЖЕННЫЕ ОБЪЕКТЫ НИЖЕ-------------------')
    for box in finalResult.boxes:
        class_id = finalResult.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        # Пушим в массив результата
        detected_elements_array.append({'name': class_id, 'cords': cords, 'conf': conf})
        # print("|-ТИП ОБЪЕКТА:", class_id)
        # print("|-КООРДИНАТЫ:", cords)
        # print("|-ВЕРОЯТНОСТЬ:", conf)
        # print("---")
    return detected_elements_array 

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
vid2 = cv2.VideoCapture(1, cv2.CAP_DSHOW) 
vid3 = cv2.VideoCapture(2, cv2.CAP_DSHOW) 
while(True): 
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    ret2, frame2 = vid2.read()
    ret3, frame3 = vid3.read()
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
    cv2.imshow('frame2', frame2)
    cv2.imshow('frame3', frame3)
    get_all_elements_from_yolo(frame)
    get_all_elements_from_yolo(frame2)
    get_all_elements_from_yolo(frame3)
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break


# After the loop release the cap object
vid.release() 
vid2.release()
vid3.release()
# Destroy all the windows 
cv2.destroyAllWindows() 

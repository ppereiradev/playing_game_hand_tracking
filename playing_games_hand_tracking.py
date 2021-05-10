import mediapipe as mp
from pykeyboard import PyKeyboard
import cv2
import math

k = PyKeyboard()


def euclidian_distance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


wCam, hCam = 800, 700

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

while True:
    _, frame = cap.read()
    image_hight, image_width, _ = frame.shape

    drawingModule = mp.solutions.drawing_utils
    handsModule = mp.solutions.hands

    with handsModule.Hands(static_image_mode=True, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(
                    frame, handLandmarks, handsModule.HAND_CONNECTIONS)

                # WRIST
                x_base = handLandmarks.landmark[handsModule.HandLandmark.WRIST].x * image_width
                y_base = handLandmarks.landmark[handsModule.HandLandmark.WRIST].y * image_hight

                # THUMB FINGER
                x_thumb_tip = handLandmarks.landmark[handsModule.HandLandmark.THUMB_TIP].x * image_width
                y_thumb_tip = handLandmarks.landmark[handsModule.HandLandmark.THUMB_TIP].y * image_hight

                x_thumb_mcp = handLandmarks.landmark[handsModule.HandLandmark.THUMB_MCP].x * image_width
                y_thumb_mcp = handLandmarks.landmark[handsModule.HandLandmark.THUMB_MCP].y * image_hight

                # INDEX FINGER
                x_index_tip = handLandmarks.landmark[handsModule.HandLandmark.INDEX_FINGER_TIP].x * image_width
                y_index_tip = handLandmarks.landmark[handsModule.HandLandmark.INDEX_FINGER_TIP].y * image_hight

                x_index_pip = handLandmarks.landmark[handsModule.HandLandmark.INDEX_FINGER_PIP].x * image_width
                y_index_pip = handLandmarks.landmark[handsModule.HandLandmark.INDEX_FINGER_PIP].y * image_hight

                # MIDDLE FINGER
                x_middle_tip = handLandmarks.landmark[handsModule.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
                y_middle_tip = handLandmarks.landmark[handsModule.HandLandmark.MIDDLE_FINGER_TIP].y * image_hight

                x_middle_pip = handLandmarks.landmark[handsModule.HandLandmark.MIDDLE_FINGER_PIP].x * image_width
                y_middle_pip = handLandmarks.landmark[handsModule.HandLandmark.MIDDLE_FINGER_PIP].y * image_hight

                # RING FINGER
                x_ring_tip = handLandmarks.landmark[handsModule.HandLandmark.RING_FINGER_TIP].x * image_width
                y_ring_tip = handLandmarks.landmark[handsModule.HandLandmark.RING_FINGER_TIP].y * image_hight

                x_ring_pip = handLandmarks.landmark[handsModule.HandLandmark.RING_FINGER_PIP].x * image_width
                y_ring_pip = handLandmarks.landmark[handsModule.HandLandmark.RING_FINGER_PIP].y * image_hight

                # PINKY FINGER
                x_pinky_tip = handLandmarks.landmark[handsModule.HandLandmark.PINKY_TIP].x * image_width
                y_pinky_tip = handLandmarks.landmark[handsModule.HandLandmark.PINKY_TIP].y * image_hight

                x_pinky_pip = handLandmarks.landmark[handsModule.HandLandmark.PINKY_PIP].x * image_width
                y_pinky_pip = handLandmarks.landmark[handsModule.HandLandmark.PINKY_PIP].y * image_hight

                # HAND IS CLOSED
                if (euclidian_distance(x_base, y_base, x_index_tip, y_index_tip) < euclidian_distance(x_base, y_base, x_index_pip, y_index_pip)) and (euclidian_distance(x_base, y_base, x_middle_tip, y_middle_tip) < euclidian_distance(x_base, y_base, x_middle_pip, y_middle_pip)) and (euclidian_distance(x_base, y_base, x_ring_tip, y_ring_tip) < euclidian_distance(x_base, y_base, x_ring_pip, y_ring_pip)) and (euclidian_distance(x_base, y_base, x_pinky_tip, y_pinky_tip) < euclidian_distance(x_base, y_base, x_pinky_pip, y_pinky_pip)):
                    k.press_key('x')

                    # if (euclidian_distance(x_pinky_pip, y_pinky_pip, x_thumb_tip, y_thumb_tip) > euclidian_distance(x_pinky_pip, y_pinky_pip, x_thumb_mcp, y_thumb_mcp)):
                    #     k.press_key('z')
                    # else:
                    #     k.release_key('z')

                # HAND IS OPEN
                elif (euclidian_distance(x_base, y_base, x_index_tip, y_index_tip) > euclidian_distance(x_base, y_base, x_index_pip, y_index_pip)) and (euclidian_distance(x_base, y_base, x_middle_tip, y_middle_tip) > euclidian_distance(x_base, y_base, x_middle_pip, y_middle_pip)) and (euclidian_distance(x_base, y_base, x_ring_tip, y_ring_tip) > euclidian_distance(x_base, y_base, x_ring_pip, y_ring_pip)) and (euclidian_distance(x_base, y_base, x_pinky_tip, y_pinky_tip) > euclidian_distance(x_base, y_base, x_pinky_pip, y_pinky_pip)):
                    k.release_key('x')
                    k.release_key('z')
                    k.release_key(k.up_key)
                    k.release_key(k.down_key)
                    k.release_key(k.right_key)
                    k.release_key(k.left_key)

                    if x_middle_tip < x_thumb_tip:
                        k.press_key(k.right_key)
                        print("x middle: ", x_middle_tip,
                              "x thumb: ", x_thumb_mcp)
                    elif x_middle_tip > x_pinky_pip:
                        print("x middle: ", x_middle_tip,
                              "x pinky: ", x_pinky_pip)
                        k.press_key(k.left_key)

                # ONLY INDEX FINGER IS UP
                elif (euclidian_distance(x_base, y_base, x_index_tip, y_index_tip) > euclidian_distance(x_base, y_base, x_index_pip, y_index_pip)) and (euclidian_distance(x_base, y_base, x_middle_tip, y_middle_tip) < euclidian_distance(x_base, y_base, x_middle_pip, y_middle_pip)) and (euclidian_distance(x_base, y_base, x_ring_tip, y_ring_tip) < euclidian_distance(x_base, y_base, x_ring_pip, y_ring_pip)) and (euclidian_distance(x_base, y_base, x_pinky_tip, y_pinky_tip) < euclidian_distance(x_base, y_base, x_pinky_pip, y_pinky_pip)):
                    k.press_key(k.up_key)
                    k.release_key('x')
                    k.release_key('z')
                    k.release_key(k.down_key)

                # ONLY PINKY IS UP
                elif (euclidian_distance(x_base, y_base, x_index_tip, y_index_tip) < euclidian_distance(x_base, y_base, x_index_pip, y_index_pip)) and (euclidian_distance(x_base, y_base, x_middle_tip, y_middle_tip) < euclidian_distance(x_base, y_base, x_middle_pip, y_middle_pip)) and (euclidian_distance(x_base, y_base, x_ring_tip, y_ring_tip) < euclidian_distance(x_base, y_base, x_ring_pip, y_ring_pip)) and (euclidian_distance(x_base, y_base, x_pinky_tip, y_pinky_tip) > euclidian_distance(x_base, y_base, x_pinky_pip, y_pinky_pip)):
                    k.press_key(k.down_key)
                    k.release_key('x')
                    k.release_key('z')
                    k.release_key(k.up_key)

                # ONLY THUMB IS UP
                elif (euclidian_distance(x_base, y_base, x_index_tip, y_index_tip) < euclidian_distance(x_base, y_base, x_index_pip, y_index_pip)) and (euclidian_distance(x_base, y_base, x_middle_tip, y_middle_tip) > euclidian_distance(x_base, y_base, x_middle_pip, y_middle_pip)) and (euclidian_distance(x_base, y_base, x_ring_tip, y_ring_tip) < euclidian_distance(x_base, y_base, x_ring_pip, y_ring_pip)) and (euclidian_distance(x_base, y_base, x_pinky_tip, y_pinky_tip) < euclidian_distance(x_base, y_base, x_pinky_pip, y_pinky_pip)):
                    k.release_key('x')
                    k.press_key('z')
                    k.release_key(k.up_key)
                    k.release_key(k.down_key)
                    k.release_key(k.right_key)
                    k.release_key(k.left_key)

    cv2.imshow("Hand", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

import cv2
import mediapipe as mp
temporary_y=590

class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.results = []

    def getFingerPosition(self, image):
        h, w, c = image.shape
        
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        fingertips = {}
        if self.results.multi_hand_landmarks:
            for hand_no, handLms in enumerate(self.results.multi_hand_landmarks):
                fingertip = []
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id in [4, 8, 12, 16, 20]:
                        cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED) #pink circle
                        fingertip.append([cx, cy])
                        cv2.line(image, (0,temporary_y), (1300,temporary_y), (255,255,255))
                        if cy > temporary_y:
                            cv2.circle(image, (cx, cy), 25, (0, 255, 0), cv2.FILLED) #green circle

                self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                fingertips[hand_no] = fingertip
        return fingertips

cap2 = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
handTracker = HandTracker()


while True:
    success2, image2 = cap2.read()

    #handTracker.getFingerPosition(image1)
    fingertips = handTracker.getFingerPosition(image2)
    print(fingertips)

    cv2.imshow("Output2", image2)

    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

cap2.release()
cv2.destroyAllWindows()
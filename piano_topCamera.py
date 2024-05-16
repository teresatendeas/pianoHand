import cv2
import numpy as np
import mediapipe as mp
import pygame
import time

pygame.init()

# Initialize sounds
sound1 = pygame.mixer.Sound("17749__jaz_the_man_2__do-re-mi-fa-so-la-ti-do/316898__jaz_the_man_2__do.wav")
sound2 = pygame.mixer.Sound("17749__jaz_the_man_2__do-re-mi-fa-so-la-ti-do/316908__jaz_the_man_2__re.wav")
sound3 = pygame.mixer.Sound("17749__jaz_the_man_2__do-re-mi-fa-so-la-ti-do/316906__jaz_the_man_2__mi.wav")
sound4 = pygame.mixer.Sound("17749__jaz_the_man_2__do-re-mi-fa-so-la-ti-do/316904__jaz_the_man_2__fa.wav")
sound5 = pygame.mixer.Sound("17749__jaz_the_man_2__do-re-mi-fa-so-la-ti-do/316912__jaz_the_man_2__sol.wav")
sound6 = pygame.mixer.Sound("17749__jaz_the_man_2__do-re-mi-fa-so-la-ti-do/316902__jaz_the_man_2__la.wav")
sound7 = pygame.mixer.Sound("17749__jaz_the_man_2__do-re-mi-fa-so-la-ti-do/316913__jaz_the_man_2__si.wav")
sound8 = pygame.mixer.Sound("17749__jaz_the_man_2__do-re-mi-fa-so-la-ti-do/316901__jaz_the_man_2__do-octave.wav")
nothing = pygame.mixer.Sound("17749__jaz_the_man_2__do-re-mi-fa-so-la-ti-do/nothing.wav")


# Play sounds simultaneously
channel1 = pygame.mixer.Channel(0)
channel2 = pygame.mixer.Channel(1)
channel3 = pygame.mixer.Channel(2)
channel4 = pygame.mixer.Channel(3)
channel5 = pygame.mixer.Channel(4)
channel6 = pygame.mixer.Channel(5)
channel7 = pygame.mixer.Channel(6)
channel8 = pygame.mixer.Channel(7)
channels = [channel1, channel2, channel3, channel4, channel5, channel6, channel7, channel8]


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
                self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                fingertips[hand_no] = fingertip
        return fingertips
# Function to get the coordinates of the center of the bounding box
def get_center_coordinates(contour):
    x, y, w, h = cv2.boundingRect(contour)
    center_x = x + w // 2
    center_y = y + h // 2
    return (center_x, center_y)

# Capture video from the default camera (camera index 0)
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
handTracker = HandTracker()

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    fingertips = handTracker.getFingerPosition(frame)

    # Define the lower and upper bounds for the green color
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a binary mask for the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area (example: exclude small contours)
    min_contour_area = 1000
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    if len(filtered_contours) >= 2:

        coordinates_x = []
        coordinates_y = []


        # Iterate through the contours
        for contour in filtered_contours:
            # Get the center coordinates of the bounding box
            center_x, center_y = get_center_coordinates(contour)
            coordinates_x.append(center_x)
            coordinates_y.append(center_y)

            # Draw a green dot at the center of the bounding box
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        #Make the list ordered ()
        if coordinates_x[0]>coordinates_x[1]:
            coordinates_x = coordinates_x[::-1]
            coordinates_y = coordinates_y[::-1]
            #print('reversed')

        #Bottom coordinates
        coordinates_x.append(coordinates_x[0])
        coordinates_x.append(coordinates_x[1])
        coordinates_y.append(coordinates_y[0]+(coordinates_x[1]-coordinates_x[0])/2.4)
        coordinates_y.append(coordinates_y[1]+(coordinates_x[1]-coordinates_x[0])/2.4)

        x_border1 = np.linspace(coordinates_x[0], coordinates_x[1], num=9)
        y_border1 = np.linspace(coordinates_y[0], coordinates_y[1], num=9)
        #(coordinates_y[1]-coordinates_y[0]) is used to straighten the x
        #x_border2 = np.linspace(coordinates_x[2], coordinates_x[3], num=9)
        x_border2 = np.linspace(coordinates_x[2]-(coordinates_y[1]-coordinates_y[0])/2.3, coordinates_x[3]-(coordinates_y[1]-coordinates_y[0])/2.3, num=9)
        y_border2 = np.linspace(coordinates_y[2], coordinates_y[3], num=9)

        points1=[]
        points2=[]
        plays=[nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing]
        
        sounds = [sound1, sound2, sound3, sound4, sound5, sound6, sound7, sound8]

        for i in range(9):
            points1.append((int(x_border1[i]), int(y_border1[i])))
            points2.append((int(x_border2[i]), int(y_border2[i])))
            cv2.circle(frame, points1[i], 5, (0, 255, 255), -1)
            cv2.circle(frame, points2[i], 5, (0, 255, 255), -1)
            cv2.line(frame, points1[i], points2[i], (0, 0, 255), 2)
        cv2.circle(frame, points1[-1], 5, (255, 255, 255), 6)

        # Print the coordinates of the center
        #print(f"X Coordinates: ({coordinates_x})")
        #print(f"Y Coordinates: ({coordinates_y})")
        for i in range(8):
            polygon_points = np.array([points1[i], points1[i+1], points2[i+1], points2[i]], np.int32)
            polygon_points = polygon_points.reshape((-1, 1, 2))
            #print(polygon_points)
            
            if fingertips is not None:
                #print('value', fingertips.values())
                for hand_points in fingertips.values():
                    for point in hand_points:
                        result = cv2.pointPolygonTest(polygon_points, point, False)
                        if result >= 0:
                            print("Point", point, "is inside the square: ", i)
                            plays[i] = sounds[i]
                channels[0].play(plays[0])
                channels[1].play(plays[1])
                channels[2].play(plays[2])
                channels[3].play(plays[3])
                channels[4].play(plays[4])
                channels[5].play(plays[5])
                channels[6].play(plays[6])
                channels[7].play(plays[7])


    # Display the resulting frame
    cv2.imshow('Green Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
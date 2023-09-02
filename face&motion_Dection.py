import cv2
import numpy as np

# Function to detect motion and faces using OpenCV
def detect_motion():
    # Open the camera (0 represents the default camera)
    cap = cv2.VideoCapture(0)
    
    # Load the Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read two initial frames to establish a baseline
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    while cap.isOpened():
        # Convert the current frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        # Calculate the absolute difference between the two frames
        diff = cv2.absdiff(frame1, frame2)
        
        # Convert the difference frame to grayscale
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply a binary threshold to create a black and white image
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        
        # Dilate the thresholded image to fill gaps
        dilated = cv2.dilate(thresh, None, iterations=3)
        
        # Find contours in the dilated image
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Iterate through the detected contours
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # Filter out small contours (noise)
            if cv2.contourArea(contour) < 1500:
                continue
            
            # Draw a rectangle around the detected motion (commented out)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add a status text for detected motion
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)
        
        # Detect faces using Haar Cascade Classifier
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
        
        # Iterate through detected faces and draw rectangles around them
        for (x, y, w, h) in faces:
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            cv2.putText(frame1, "Status: {}".format('Face Found'), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)
        
        # Display the frame with motion detection and face detection
        cv2.imshow("Motion Detection", frame1)
        
        # Update the previous frame
        frame1 = frame2
        
        # Read the next frame from the camera
        ret, frame2 = cap.read()
        
        # Check if the 'q' key is pressed to quit the application
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cv2.destroyAllWindows()
    cap.release()

# Entry point of the script
if __name__ == "__main__":
    detect_motion()

from flask import Flask, render_template, Response, request
import cv2
import numpy as np

app = Flask(__name__, 
            static_url_path='/static',
            static_folder='static')

# Initialize the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def calculate_distance(face_width, known_width=16.0):
    focal_length = 600
    distance = (known_width * focal_length) / face_width
    return round(distance, 1)

def get_emotion(gray, x, y, w, h):
    roi_gray = gray[y:y+h, x:x+w]
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
    
    # Detect smile
    smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
    
    # Calculate region brightness
    brightness = np.mean(roi_gray)
    
    # Determine emotion based on features
    if len(smile) > 0:
        if len(eyes) >= 2:  # Both eyes visible
            emotion = "Happy"
            confidence = min(100, 70 + len(smile) * 10)
        else:  # Eyes might be closed - could be laughing
            emotion = "Very Happy"
            confidence = min(100, 80 + len(smile) * 10)
    else:
        if brightness < 100:
            emotion = "Sad"
            confidence = min(100, (1 - brightness/100) * 100)
        elif len(eyes) >= 2:
            emotion = "Neutral"
            confidence = 70
        else:
            emotion = "Surprised"
            confidence = 65
    
    return emotion, confidence

def generate_frames_face_only():
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 155, 255), 2)
            
            # Calculate distance
            distance = calculate_distance(w)
            
            # Create background for text
            cv2.rectangle(frame, (x, y-60), (x + 200, y), (0, 155, 255), -1)
            
            # Add text
            text = f"Face Detected"
            distance_text = f"Distance: {distance}cm"
            
            cv2.putText(frame, text, (x+5, y-35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, distance_text, (x+5, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_with_emotions():
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 155, 255), 2)
            
            try:
                # Get emotion
                emotion, confidence = get_emotion(gray, x, y, w, h)
                
                # Create background for text
                cv2.rectangle(frame, (x, y-60), (x + 250, y), (0, 155, 255), -1)
                
                # Add text
                emotion_text = f"Emotion: {emotion}"
                confidence_text = f"Confidence: {confidence:.2f}%"
                
                cv2.putText(frame, emotion_text, (x+5, y-35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, confidence_text, (x+5, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Error in emotion detection: {e}")
                
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_eye_detection():
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            
            # Draw rectangle for face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 155, 255), 2)
            
            # Count detected eyes
            eye_count = len(eyes)
            
            # Create background for text
            cv2.rectangle(frame, (x, y-60), (x + 200, y), (0, 155, 255), -1)
            
            # Add text for eye count
            status_text = f"Eyes Detected: {eye_count}"
            cv2.putText(frame, status_text, (x+5, y-25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw rectangles around eyes
            for (ex, ey, ew, eh) in eyes:
                # Convert relative coordinates to absolute
                abs_x = x + ex
                abs_y = y + ey
                
                # Draw rectangle around each eye
                cv2.rectangle(frame, (abs_x, abs_y), (abs_x+ew, abs_y+eh), (255, 255, 0), 2)
                
                # Add "Eye Detected" text above each eye
                cv2.putText(frame, "Eye", (abs_x, abs_y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face-detection')
def face_detection():
    return render_template('face_detection.html')

@app.route('/emotion-detection')
def emotion_detection():
    return render_template('emotion_detection.html')

@app.route('/body-detection')
def body_detection():
    return render_template('coming_soon.html',
                         page_name="Body Detection",
                         icon_class="fas fa-walking",
                         feature_name="body detection",
                         features=[
                             "Full body pose estimation",
                             "Real-time movement tracking",
                             "Multiple person detection",
                             "Gesture recognition",
                             "Activity classification"
                         ])

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_face_only(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_video_feed')
def emotion_video_feed():
    return Response(generate_frames_with_emotions(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/eye-detection')
def eye_detection():
    return render_template('eye_detection.html')

@app.route('/eye_video_feed')
def eye_video_feed():
    return Response(generate_frames_eye_detection(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_sensitivity')
def update_sensitivity():
    sensitivity = float(request.args.get('value', 1.1))
    # Update the face detection parameters
    return {'status': 'success'}

@app.route('/toggle_distance')
def toggle_distance():
    show_distance = request.args.get('show', 'true') == 'true'
    # Update the distance display setting
    return {'status': 'success'}

if __name__ == "__main__":
    app.run(debug=True) 
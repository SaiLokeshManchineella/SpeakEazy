# Step 1: Import necessary libraries
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from time import time
def method():
    actions = np.array(['1. loud','2. quiet','3. happy','4. sad','5. Beautiful','6. Ugly','7. Deaf','8. Blind','9. Nice','10. Mean','11. rich','12. poor','13. thick','14. thin','15. expensive','16. cheap','17. flat','18. curved','19. male','20. female','21. tight','22. loose','23. high','24. low','25. soft','26. hard','27. deep','28. shallow','29. clean','30. dirty','31. strong','32. weak','33. dead','34. alive','35. heavy'])
    # Step 2: Load the .h5 model
    model = load_model("action75a.h5")

    # Step 3: Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Step 4: Create a loop to continuously capture frames from the webcam
    t=time()
    while time()-t<40:
        # Step 5: Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break
        
        # Step 5: Preprocess each frame as required by your sign language recognition model
    # Ensure this matches your model's training preprocessing
    # Example preprocessing (modify as needed):
        frame_resized = cv2.resize(frame, (1662, 40)) # Resize to model's expected input size
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY) # Convert to grayscale if needed
        processed_frame = np.expand_dims(frame_gray, axis=-1) # Add channel dimension if needed
        processed_frame = np.expand_dims(processed_frame, axis=0) # Add batch dimension

    # Step 6: Pass the preprocessed frame to the loaded .h5 model to predict
        prediction = model.predict(processed_frame)
        predicted_class_index = np.argmax(prediction[0])
        class_name = actions[predicted_class_index].split()
    
        # Modify this section of your code to match your model's expected input shape
        processed_frame = cv2.resize(frame, (1662, 40)) # Resize to model's expected input size
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale if your model expects one channel
        processed_frame = np.expand_dims(processed_frame, axis=-1) # Add channel dimension if needed
        processed_frame = np.expand_dims(processed_frame, axis=0) # Add batch dimension

        
        # Step 7: Display the prediction on each video frame
        cv2.putText(frame, 
                    'Prediction: {}'.format(class_name[1]), 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255,255,255), 
                    2)

        # Step 8: Show the modified video frames in real-time
        cv2.imshow('Sign Language Recognition', frame)

        # Step 9: Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    method()
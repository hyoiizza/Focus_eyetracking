import cv2
import numpy as np

def sobel_gradient(image):
    """
    Compute the Sobel gradient magnitude of the input image.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude

def ray_cast_with_sobel(frame, num_rays=360, max_radius=100, step_size=1):
    """
    Perform ray casting with Sobel edge detection on a single frame.
    """
    # Convert to grayscale and preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_mag = sobel_gradient(blurred)

    # Define initial center (middle of the frame)
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    edges = []

    # Cast rays
    for i in range(num_rays):
        theta = 2 * np.pi * i / num_rays
        for r in range(0, max_radius, step_size):
            x = int(center_x + r * np.cos(theta))
            y = int(center_y + r * np.sin(theta))

            # Check if coordinates are within bounds
            if x < 0 or x >= width or y < 0 or y >= height:
                break

            # Check Sobel gradient magnitude
            if sobel_mag[y, x] > 50:  # Adjust threshold as needed
                edges.append((x, y))
                break  # Stop at the first edge detected along this ray

    return edges

def process_video(video_source=0):
    """
    Process video input with ray casting and display results.
    """
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        # Apply ray casting
        edges = ray_cast_with_sobel(frame)

        # Draw detected edges
        for x, y in edges:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display the processed frame
        cv2.imshow("Ray Casting with Sobel", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the video processing function
process_video('9.avi')  # Use 0 for webcam or replace with a video file path


'''
difference between 11월 27일 재구성 / 11월 29일 

소벨 연산자 적용 - 윤곽선 
....흠 다시 제대로 읽어봐야곘다


'''

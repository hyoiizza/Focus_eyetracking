import numpy as np
import cv2
import pyautogui
import time
from screeninfo import get_monitors
import sys


'''
starburst + gui complete
그러나 스타버스트 코드는 수정전 버전 코드임.
'''

class Star:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Failed to open the video file.")
        
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def preprocess_frame(self):
        """Preprocess the video frame for pupil detection."""
        ret, frame = self.cap.read()
        if not ret:
            print("No more frames to read.")
            return None
        self.frame = frame
        bright_frame = cv2.convertScaleAbs(self.frame, alpha=1.5, beta=0)
        self.gray = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2GRAY)
        return self.frame

    def ray_casting(self, num_rays=360, max_radius=100, step_size=1):
        """Perform ray casting to detect edge points."""
        self.xc, self.yc = self.video_width // 2, self.video_height // 2
        edges = []
        _, threshold = cv2.threshold(self.gray, 100, 255, cv2.THRESH_BINARY_INV)
        sobel_x = cv2.Sobel(threshold, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(threshold, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        for i in range(num_rays):
            theta = 2 * np.pi * i / num_rays
            for r in range(0, max_radius, step_size):
                x = int(self.xc + r * np.cos(theta))
                y = int(self.yc + r * np.sin(theta))

                if x < 0 or x >= self.video_width or y < 0 or y >= self.video_height:
                    break
                if sobel_mag[y, x] > 0:
                    edges.append((x, y))
                    break
        return edges

    def ransac_apply(self, points, sample_size=5, max_iterations=1000, stop_inlier_ratio=0.7):
        """Fit an ellipse to the edge points using RANSAC."""
        best_ellipse = None
        max_inliers = 0
        points = np.array(points, dtype=np.float32)

        if len(points) < sample_size:
            print("Not enough points for ellipse fitting.")
            return None

        for _ in range(max_iterations):
            if len(points) < sample_size:
                print("Not enough points for ellipse fitting.")
                break
            sample = points[np.random.choice(len(points), sample_size, replace=False)]
            try:
                if len(sample) >= 5:
                    ellipse = cv2.fitEllipse(sample)
                    if np.isnan(ellipse[1][0]) or np.isnan(ellipse[1][1]):
                        continue
                    center, axes, angle = ellipse
                    inliers = []
                    for point in points:
                        distance = cv2.pointPolygonTest(
                        cv2.ellipse2Poly(
                            center=(int(center[0]), int(center[1])),
                            axes=(int(axes[0] / 2), int(axes[1] / 2)),
                            angle=int(angle),
                            arcStart=0,
                            arcEnd=360,
                            delta=5,
                        ),
                        tuple(point),
                        True,
                    )
                    if abs(distance) <= 5:  # 인라이어 거리 임계값
                        inliers.append(point)

                    if len(inliers) > max_inliers:
                        max_inliers = len(inliers)
                        best_ellipse = ellipse
            except cv2.error:
                continue

            if max_inliers / len(points) >= stop_inlier_ratio:
                break
        return best_ellipse

    def draw_ellipse(self, ellipse):
        """Draw the fitted ellipse on the frame."""
        if ellipse is not None:
            cv2.ellipse(self.frame, ellipse, (0, 255, 0), 2)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            print(f"Ellipse center: {center}")
        else:
            print("No ellipse detected.")
        return ellipse

    def release_resources(self):
        """Release video capture resources."""
        self.cap.release()
        cv2.destroyAllWindows()


class GUI(Star):
    def __init__(self, video_path):
        super().__init__(video_path)
        monitor = get_monitors()[0]
        self.screen_width = monitor.width
        self.screen_height = monitor.height

    def norm_coord(self, x, y):
        """Normalize video coordinates to screen coordinates."""
        norm_x = max(0, min(x / self.video_width,1))
        norm_y = max(0,min(y / self.video_height,1))
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)
        return screen_x, screen_y

    def move_mouse_to_center(self, center):
        """Move the mouse to the detected ellipse center."""
        if center is not None:
            screen_x, screen_y = self.norm_coord(center[0], center[1])
            screen_x = max(1, min(screen_x,pyautogui.size().width -1))
            screen_y = max(1,min(screen_y,pyautogui.size().height -1))
            pyautogui.moveTo(screen_x, screen_y)
            pyautogui.click()
            print(f"Mouse moved and clicked at ({screen_x}, {screen_y}).")


if __name__ == "__main__":
    a = GUI("csub.mp4")

    while True:
        frame = a.preprocess_frame()
        if frame is None:
            break
        edges = a.ray_casting()
        best_ellipse = a.ransac_apply(edges)
        a.draw_ellipse(best_ellipse)
        if best_ellipse:
            center = best_ellipse[0]
            a.move_mouse_to_center(center)

        cv2.imshow("Video", a.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    a.release_resources()

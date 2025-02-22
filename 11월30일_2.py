import numpy as np
import cv2
import pyautogui
import time
from screeninfo import get_monitors
import sys


'''
작동 안함 왜지? 

'''
class Star:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            print("Failed to read the video.")
            sys.exit(1)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.bright_frame = cv2.convertScaleAbs(self.frame, alpha=1.5, beta=0)
        self.gray = cv2.cvtColor(self.bright_frame, cv2.COLOR_BGR2GRAY)

    def ray_casting(self, num_rays=360, max_radius=100, step_size=1):
        xc, yc = self.video_width // 2, self.video_height // 2
        edges = []

        _, threshold = cv2.threshold(self.gray, 100, 255, cv2.THRESH_BINARY_INV)

        sobel_x = cv2.Sobel(threshold, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(threshold, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        for i in range(num_rays):
            theta = 2 * np.pi * i / num_rays

            for r in range(0, max_radius, step_size):
                x = int(xc + r * np.cos(theta))
                y = int(yc + r * np.sin(theta))

                if x < 0 or x >= self.frame.shape[1] or y < 0 or y >= self.frame.shape[0]:
                    break

                sobel_value = sobel_mag[y, x] if (0 <= x < sobel_mag.shape[1] and 0 <= y < sobel_mag.shape[0]) else 0
                if sobel_value > 0:
                    edges.append((x, y))
                    break

        return edges

    def ransac_apply(self, points, sample_size=5, min_iterations=20, max_iterations=1000, stop_inlier_ratio=0.7):
        best_ellipse = None
        max_inliers = 0
        points = np.array(points, dtype=np.float32)

        if len(points) < 5:
            print("At least 5 points are needed.")
            return None

        for _ in range(max_iterations):
            sample = points[np.random.choice(len(points), sample_size, replace=False)]

            try:
                if len(sample) >= 5:
                    ellipse = cv2.fitEllipse(sample)
                    inliers = [point for point in points if cv2.pointPolygonTest(ellipse, tuple(point), True) <= 5]

                    if len(inliers) > max_inliers:
                        max_inliers = len(inliers)
                        best_ellipse = ellipse

            except cv2.error:
                continue

            if max_inliers / len(points) >= stop_inlier_ratio:
                break

        return best_ellipse

    def draw_ellipse(self, best_ellipse):
        if best_ellipse is not None:
            cv2.ellipse(self.frame, best_ellipse, (0, 255, 0), 2)
            cv2.imshow("Detected Ellipse", self.frame)
        else:
            print("No ellipse detected")


class GUI(Star):
    def __init__(self, video_path):
        super().__init__(video_path)

    @staticmethod
    def norm_coord(x, y, video_width, video_height):
        monitor = get_monitors()[0]
        screen_width = monitor.width
        screen_height = monitor.height

        norm_x = x / video_width
        norm_y = y / video_height

        screen_x = int(norm_x * screen_width)
        screen_y = int(norm_y * screen_height)

        return screen_x, screen_y

    def move_command_mouse(self, center):
        if center:
            x, y = center
            screen_x, screen_y = self.norm_coord(x, y, self.video_width, self.video_height)
            pyautogui.moveTo(screen_x, screen_y)
            pyautogui.click()
            print(f"Mouse moved to and clicked at ({screen_x}, {screen_y})")

    def run(self):
        while True:
            ray_cast_result = self.ray_casting()
            ransac_result = self.ransac_apply(ray_cast_result)
            if ransac_result:
                center, _, _ = ransac_result
                self.move_command_mouse(center)
            self.draw_ellipse(ransac_result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    gui_app = GUI("csub.mp4")
    gui_app.run()

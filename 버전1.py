import numpy as np
import cv2
import pyautogui
from screeninfo import get_monitors
import tkinter as tk
from PIL import Image, ImageTk


'''
Gui + Starburst + click ver

< 값 변경 가능 변수 >
1) line 39 : _, self.threshold = cv2.threshold(self.gray, 50, 255, cv2.THRESH_BINARY) 에서 '50' 
● 픽셀값이 50이상이면, 255(흰색)으로 설정
● 픽셀값이 50이하면, 0(검정)으로 설정
이진화 임계값 조정가능 - 영상이 어두울 수록 값을 낮춰야함 
2) line 41 &154 : max_radius=100 영상,눈크기에 따라 값 조정해야함
3) line 67 : max_iterations=300 
4) line 86 : if ellipse[1][0] > 40 or ellipse[1][1] > 40: 타원의 주축, 부축 길이 40 이하인 타원만 인지하도록 조정한것 : 검출되는 타원의 크기에 맞춰 조정가능 


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

    def ray_casting(self, num_rays=360, max_radius=100):
        """Perform ray casting to detect edge points."""
        self.xc, self.yc = self.video_width // 2, self.video_height // 2
        edges = []
        _, self.threshold = cv2.threshold(self.gray, 50, 255, cv2.THRESH_BINARY)
        sobel_x = cv2.Sobel(self.threshold, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.threshold, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        for i in range(num_rays):
            theta = 2 * np.pi * i / num_rays
            for r in range(0, max_radius):
                x = int(self.xc + r * np.cos(theta))
                y = int(self.yc + r * np.sin(theta))

                if x < 0 or x >= self.video_width or y < 0 or y >= self.video_height:
                    break
                if sobel_mag[int(y), int(x)] > 50:  # 50은 임의로 설정된 값
                    edges.append((x, y))
                    
        return edges

    def ransac_apply(self, points, sample_size=5, max_iterations=1000 ,inlier_threshold=3,stop_inlier_ratio=0.7):
        """Fit an ellipse to the edge points using RANSAC."""
        best_ellipse = None
        max_inliers = 0
        points = np.array(points, dtype=np.float32)

        if len(points) < sample_size:
            print("Not enough points for ellipse fitting.")
            return None

        for _ in range(max_iterations):
            sample = points[np.random.choice(len(points), sample_size, replace=False)]
            try:
                if len(sample) >= 5:
                    ellipse = cv2.fitEllipse(sample)
                    
                    if ellipse[1][0] > 40 or ellipse[1][1] > 40:
                        continue
                    
                    if np.isnan(ellipse[1][0]) or np.isnan(ellipse[1][1]):
                        continue
                    center, axes, angle = ellipse
                    inliers = []
                    distances= []
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
                        if abs(distance) <= inlier_threshold:  # Inlier distance threshold
                            inliers.append(point)
                            distances.append(int(distance))
                    #print(f"  Distances of inliers: {distances}")
                    if len(inliers) > max_inliers:
                        max_inliers = len(inliers)
                        best_ellipse = ellipse
            except cv2.error:
                continue

            inlier_ratio = max_inliers / len(points)
            if inlier_ratio >= stop_inlier_ratio:
                print('조기 종료 조건 만족')
                break

        return best_ellipse

    def draw_ellipse(self, ellipse):
        """Draw the fitted ellipse on the image."""
        if ellipse is not None:
            center, axes, angle = ellipse
            print(f"타원의 중심: {center}, 주축,부축 길이: {axes}, 회전 각도: {angle}")
            # 타원의 넓이 계산
            a = axes[0] / 2  # 반장축 (semi-major axis)
            b = axes[1] / 2  # 반단축 (semi-minor axis)
            area = np.pi * a * b
            print(f"타원의 넓이: {area:.2f}")
            cv2.ellipse(self.frame, ellipse, (0, 0, 255), 2)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.circle(self.frame, (int(ellipse[0][0]), int(ellipse[0][1])), 2, (255, 0, 255), -1)
            print(f"Ellipse center: {center}")
        else:
            print("No ellipse detected.")
        
    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()


class Click(Star):
    def __init__(self, video_path):
        super().__init__(video_path)
        monitor = get_monitors()[0]
        self.screen_width = monitor.width
        self.screen_height = monitor.height

    def norm_coord(self, x, y):
        norm_x = max(0, min(x / self.video_width, 1))
        norm_y = max(0, min(y / self.video_height, 1))
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)
        return screen_x, screen_y

    def move_mouse_to_center(self, center):
        if center is not None:
            screen_x, screen_y = self.norm_coord(center[0], center[1])
            screen_x = max(1, min(screen_x,pyautogui.size().width -10))
            screen_y = max(1,min(screen_y,pyautogui.size().height -10))
            pyautogui.moveTo(screen_x, screen_y)
            pyautogui.click()
            print(f"Mouse moved and clicked at ({screen_x}, {screen_y}).")


class VideoApp(tk.Tk):
    def __init__(self, video_path):
        super().__init__()
        self.title("Video GUI")
        self.geometry("400x300")
        self.wm_attributes("-topmost", 1)

        # OpenCV 및 Click 클래스 초기화
        self.click = Click(video_path)

        # 영상 표시를 위한 Label 위젯 생성
        self.video_label = tk.Label(self)
        self.video_label.pack()

        # 주기적 업데이트
        self.update_video()

    def update_video(self):
        """Update the video frame in the GUI."""
        frame = self.click.preprocess_frame()
        if frame is None:
            self.click.release_resources()
            self.destroy()
            return

        # 엣지와 타원 검출
        edges = self.click.ray_casting(num_rays=360, max_radius=100)
        best_ellipse = self.click.ransac_apply(edges)
        self.click.draw_ellipse(best_ellipse)
        if best_ellipse:
            self.click.move_mouse_to_center(best_ellipse[0])

        frame = cv2.cvtColor(self.click.frame, cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.video_label.config(image=image)
        self.video_label.image = image

        # 30ms 후 재호출
        self.after(30, self.update_video)


if __name__ == "__main__":
    video_path = "Time2.mov"
    a = VideoApp(video_path)
    a.mainloop()

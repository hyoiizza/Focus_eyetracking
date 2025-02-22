import numpy as np
import cv2
import ctypes
import sys
import pyautogui
import time
from screeninfo import get_monitors


'''
스타버스트 + condensation 렌섹대신임 video 
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
        self.xc, self.yc = self.video_width // 2, self.video_height // 2
        edges = []
        _, self.threshold = cv2.threshold(self.gray, 50, 255, cv2.THRESH_BINARY)
        self.sobel_x = cv2.Sobel(self.threshold, cv2.CV_64F, 1, 0, ksize=3)
        self.sobel_y = cv2.Sobel(self.threshold, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(self.sobel_x**2 + self.sobel_y**2)

        for i in range(num_rays):
            theta = 2 * np.pi * i / num_rays

            for r in range(0, max_radius):
                x = int(self.xc + r * np.cos(theta))
                y = int(self.yc + r * np.sin(theta))
                if x < 0 or x >= self.frame.shape[1] or y < 0 or y >= self.frame.shape[0]:
                    break
                sobel_value = sobel_mag[int(y), int(x)] if (0 <= x < sobel_mag.shape[1] and 0 <= y < sobel_mag.shape[0]) else 0
                if sobel_value > 70:  # Increase threshold to detect more edges
                    edges.append((x, y))
        return edges

    def condensation_apply(self, num_particles=60, max_iterations=300, stop_inlier_ratio=0.7):
        # 입자 필터 초기화
        particles = []
        for _ in range(num_particles):
            # 초기 입자들: 타원의 중심과 축을 랜덤하게 초기화
            center = (np.random.randint(0, self.video_width), np.random.randint(0, self.video_height))
            axes = (np.random.randint(20, 50), np.random.randint(20, 50))
            angle = np.random.randint(0, 360)
            particles.append({'center': center, 'axes': axes, 'angle': angle, 'weight': 1.0})

        best_ellipse = None
        max_weight = 0

        for _ in range(max_iterations):
            # 입자 예측
            for particle in particles:
                # 입자에 대해 밀도 평가하기 위한 타원을 그린다.
                center, axes, angle = particle['center'], particle['axes'], particle['angle']
                # 타원의 포인트가 아닌 중심, 축, 각도를 calculate_density로 전달
                density = self.calculate_density((center, axes, angle), edges)
                
                # 타원의 밀도를 가중치로 설정
                particle['weight'] = density  # 밀도를 직접 가중치로 사용
            
            # 리샘플링
            particles = self.resample_particles(particles)
            
            # 가장 높은 가중치의 입자를 찾는다.
            best_particle = max(particles, key=lambda p: p['weight'])
            if best_particle['weight'] > max_weight:
                max_weight = best_particle['weight']
                best_ellipse = (best_particle['center'], best_particle['axes'], best_particle['angle'])

            # 가중치 비율이 충분히 높으면 종료
            if max_weight / len(particles) >= stop_inlier_ratio:
                break

        return best_ellipse


    def calculate_density(self, ellipse, edges):
        # ellipse는 (center, axes, angle) 형태여야 함
        if isinstance(ellipse, tuple) and len(ellipse) == 3:
            center, axes, angle = ellipse
        else:
            raise ValueError("Ellipse should be a tuple of (center, axes, angle).")

        # 타원의 반장축(a)과 반단축(b)
        a = axes[0] / 2
        b = axes[1] / 2
        delta = 5  # 경계 근처의 두께 설정

        # 엣지 점 집합에서 타원 방정식 기반 밀도 계산
        cos_theta = np.cos(np.radians(angle))
        sin_theta = np.sin(np.radians(angle))
        
        # 타원의 경계 계산을 위한 좌표 변환
        def is_near_boundary(x, y):
            x_prime = cos_theta * (x - center[0]) + sin_theta * (y - center[1])
            y_prime = -sin_theta * (x - center[0]) + cos_theta * (y - center[1])
            value = (x_prime**2 / a**2) + (y_prime**2 / b**2)
            return 1 - delta <= value <= 1 + delta

        # 엣지 점 중 타원 경계 근처 점 찾기
        boundary_edges = [edge for edge in edges if is_near_boundary(edge[0], edge[1])]

        # 밀도 계산 (정규화)
        density = len(boundary_edges) / (2 * np.pi * np.sqrt(a**2 + b**2))  # 근사 경계 길이로 정규화
        return density


    def resample_particles(self, particles):
        # 리샘플링: 가중치가 높은 입자들로부터 새로운 입자 세트를 만든다.
        weights = [p['weight'] for p in particles]
        total_weight = sum(weights)
        if total_weight == 0:
            return particles  # 가중치가 모두 0일 경우 리샘플링 없이 그대로 반환
        probabilities = [w / total_weight for w in weights]
        indices = np.random.choice(len(particles), size=len(particles), p=probabilities)
        resampled_particles = [particles[i] for i in indices]
        return resampled_particles

    def draw_ellipse(self, ellipse):
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


if __name__ == "__main__":
    a = Star("ir4.avi")

    while True:
        frame = a.preprocess_frame()
        if frame is None:
            break
        edges = a.ray_casting()
        best_ellipse = a.condensation_apply(num_particles=100, max_iterations=300)
        a.draw_ellipse(best_ellipse)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    a.release_resources()

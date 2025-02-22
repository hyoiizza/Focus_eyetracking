import numpy as np
import cv2
import ctypes
import sys
import pyautogui
import time
from screeninfo import get_monitors

'''
스타버스트 + condensation 렌섹대신임 pic
'''

class Star:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Failed to open the image file.")
        
        self.video_width = self.image.shape[1]
        self.video_height = self.image.shape[0]

    def preprocess_frame(self):
        """Preprocess the image for pupil detection."""
        bright_frame = cv2.convertScaleAbs(self.image, alpha=1.5, beta=0)
        self.gray = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2GRAY)
        return self.image
    
    def ray_casting(self, num_rays=360, max_radius=450):
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
                if x < 0 or x >= self.image.shape[1] or y < 0 or y >= self.image.shape[0]:
                    break
                sobel_value = sobel_mag[int(y), int(x)] if (0 <= x < sobel_mag.shape[1] and 0 <= y < sobel_mag.shape[0]) else 0
                if sobel_value > 70:  # Increase threshold to detect more edges
                    edges.append((x, y))
        return edges

    def condensation_apply(self, num_particles=50, max_iterations=300, stop_inlier_ratio=0.6):
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
                inside_density, boundary_density = self.calculate_density((center, axes, angle), edges, self.image)
                
                # 타원의 밀도를 가중치로 설정
                particle['weight'] = inside_density + boundary_density  # We can combine both densities or choose one
             
            
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


    def calculate_density(self, ellipse, edges, frame):
        # ellipse는 (center, axes, angle) 형태여야 함
        if isinstance(ellipse, tuple) and len(ellipse) == 3:
            center, axes, angle = ellipse
        else:
            raise ValueError("Ellipse should be a tuple of (center, axes, angle).")

        # 타원의 반장축(a)과 반단축(b)
        a = axes[0] / 2
        b = axes[1] / 2
        ellipse_area = np.pi * a * b  # 타원의 면적
        
        # 타원 경계의 길이 계산 (Ramanujan의 근사식)
        ellipse_perimeter = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

        # 타원 경계의 다각형을 생성
        ellipse_points = cv2.ellipse2Poly(center=(int(center[0]), int(center[1])), 
                                        axes=(int(axes[0] / 2), int(axes[1] / 2)), 
                                        angle=int(angle), arcStart=0, arcEnd=360, delta=5)
        
        # 타원 내부 및 경계를 계산할 리스트
        inside_edges = []
        boundary_edges = []

        # 엣지가 타원 내부에 포함되는지 확인
        for edge in edges:
            x, y = edge
            dist = cv2.pointPolygonTest(ellipse_points, (x, y), True)
            
            # 경계와의 거리 조건 (예: 3픽셀 이내)
            if abs(dist) <= 3:
                boundary_edges.append((x, y))  # 경계에 가까운 점
            elif dist > 3:  # 타원 내부에 있는 점
                inside_edges.append((x, y))
        
        # 밀도 계산
        inside_density = len(inside_edges) / ellipse_area  # 내부 밀도
        boundary_density = len(boundary_edges) / ellipse_perimeter  # 경계 밀도
        
        return inside_density, boundary_density

    def resample_particles(self, particles):
        # 리샘플링: 가중치가 높은 입자들로부터 새로운 입자 세트를 만든다.
        weights = [p['weight'] for p in particles]
        total_weight = sum(weights)
        if total_weight == 0:
            return particles  # 가중치가 모두 0일 경우 리샘플링 없이 그대로 반환
        probabilities = [w / total_weight for w in weights]
        resampled_particles = np.random.choice(particles, len(particles), p=probabilities)
        return resampled_particles.tolist()

    def draw_ellipse(self, ellipse):
        if ellipse is not None:
            center, axes, angle = ellipse
            print(f"타원의 중심: {center}, 주축,부축 길이: {axes}, 회전 각도: {angle}")
            # 타원의 넓이 계산
            a = axes[0] / 2  # 반장축 (semi-major axis)
            b = axes[1] / 2  # 반단축 (semi-minor axis)
            area = np.pi * a * b
            print(f"타원의 넓이: {area:.2f}")
            cv2.ellipse(self.image, ellipse, (0, 0, 255), 2)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.circle(self.image, (int(ellipse[0][0]), int(ellipse[0][1])), 2, (255, 0, 255), -1)
            print(f"Ellipse center: {center}")
        else:
            print("No ellipse detected.")
    
    def release_resources(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    a = Star("ir4-5.png")  # 이미지 경로로 수정

    frame = a.preprocess_frame()
    edges = a.ray_casting()
    best_ellipse = a.condensation_apply(num_particles=100, max_iterations=300)
    a.draw_ellipse(best_ellipse)
    cv2.imshow("Processed Image", frame)
    cv2.waitKey(0)  # 사용자 키 입력을 기다림
    a.release_resources()

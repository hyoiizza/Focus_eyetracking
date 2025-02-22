import numpy as np
import cv2
import random
import math


'''
star + ransac(밀도로 적합도평가) : pic
1) max_radius 가상광선길이 /작은 영상=100, 큰영상=450
2) thresh 이진화 임계값 / 일반적으로 50이면 적당하나 영상이 어두우면 20
3) threshold=3 가까운점인지 인식할 기준 임계값 / 일반적으로 3
4)min_density_threshold=1 밀도 임계값 / 일반적으로 1 
5) min_length 타원 최소 주축,부축 길이 일반적으로 40이나 영상이 작으면 20 / 20-40
6) max_length 타원 최대 주축,부축 길이 일반적으로 150이나 영상이 작으면 80, 영상이 크면 180/ 80-180
'''

class Star:
    def __init__(self, image_path):
        # 이미지 로딩
        self.frame = cv2.imread(image_path)
        if self.frame is None:
            raise ValueError("Failed to load the image.")
        
        self.image_width = self.frame.shape[1]
        self.image_height = self.frame.shape[0]
        self.xc, self.yc = self.image_width // 2, self.image_height // 2

    def preprocess_frame(self):
        # 이미지를 밝게 하고 회색조로 변환
        bright_frame = cv2.convertScaleAbs(self.frame, alpha=1.5, beta=0)
        self.gray = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2GRAY)
        return self.gray

    def ray_casting(self, num_rays=360, max_radius=100,thresh=50):
        edges = []
        _, self.threshold = cv2.threshold(self.gray, thresh, 255, cv2.THRESH_BINARY)
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
        for x, y in edges:
            cv2.circle(self.frame, (x, y), 1, (0, 255, 0), -1)
        print('total edges:',len(edges),'개')
        return edges

    def fit_ellipse_ransac(self, edges, num_iterations=500, threshold=3,min_density_threshold=1,min_length=20,max_length=40):
        # RANSAC 알고리즘을 이용한 타원 추정
        best_ellipse = None
        best_density = 0
        best_inliers=[]
        for _ in range(num_iterations):
            # 무작위로 5개의 엣지 점 선택
            sample_points = random.sample(edges,5)
            
            # 타원 파라미터 추정
            ellipse = self.fit_ellipse_from_points(sample_points)
            if ellipse[1][0] < min_length or ellipse[1][1] < min_length:
                print('ellipse too small')
                continue
            if  ellipse[1][0] > max_length or ellipse[1][1] > max_length:
                print('ellipse too big')
                continue
            
            # 타원에 대한 인라이어 계산
            inliers = self.get_inliers(edges, ellipse, threshold)
            
            # 밀도 계산
            density = self.calculate_density(ellipse, inliers)

            # 최적의 타원 선택 (밀도 기준)
            if density > min_density_threshold:
                best_density = density
                best_ellipse = ellipse
                best_inliers = inliers
                
        print('inliers: ', len(best_inliers))
        print('최적 타원의 밀도:', best_density)
        
        return best_ellipse

    def fit_ellipse_from_points(self, points):
        # 최소 5개의 점으로 타원 파라미터 추정
        points = np.array(points)
        ellipse = cv2.fitEllipse(points)
        return ellipse

    def get_inliers(self, edges, ellipse, threshold):
        # 타원에 가까운 점들을 인라이어로 간주
        inliers = []
        for x, y in edges:
            if self.is_point_near_ellipse(x, y, ellipse, threshold):
                inliers.append((x, y))
        
        return inliers

    def is_point_near_ellipse(self, x, y, ellipse, threshold):
        # 점이 타원에 가까운지 확인하는 함수
        cx, cy = ellipse[0]
        a, b = ellipse[1]
        angle = ellipse[2]

        dx = (x - cx) * np.cos(np.radians(angle)) + (y - cy) * np.sin(np.radians(angle))
        dy = -(x - cx) * np.sin(np.radians(angle)) + (y - cy) * np.cos(np.radians(angle))

        distance = (dx**2) / (a**2) + (dy**2) / (b**2)
        # print(f"점 ({x}, {y})가 타원에 가까운지 확인: {abs(distance - 1)} < {threshold}")
        return abs(distance - 1) < threshold

    def calculate_density(self, ellipse, inliers):
        # 타원의 밀도 계산
        a, b = ellipse[1]
        perimeter = np.pi * (3 * (a + b) - np.sqrt((3*a + b) * (a + 3*b)))
        density = len(inliers) / perimeter
        return density

    def draw_ellipse(self, ellipse, edges):
        # 타원과 경계 근처 점들 시각화
        if ellipse is not None:
            center, axes, angle = ellipse
            print(f"타원의 중심: {center}, 주축, 부축 길이: {axes}, 회전 각도: {angle}")
            a = axes[0] / 2
            b = axes[1] / 2
            area = np.pi * a * b
            print(f"타원의 넓이: {area:.2f}")
            C = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
            print(f"타원 경계의 길이: {C:.2f}")
            
            # 경계 근처 점들 시각화
            for edge in edges:
                if self.is_point_near_ellipse(edge[0], edge[1], ellipse, threshold=3):
                    cv2.circle(self.frame, (edge[0], edge[1]), 1, (0, 255, 0), -1)  # 초록색 원으로 표시
            #타원 시각화 
            cv2.ellipse(self.frame, ellipse, (0, 0, 255), 3)
            #타원중심시각화
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.circle(self.frame, (int(ellipse[0][0]), int(ellipse[0][1])), 4, (0, 0,255), -1)
        else:
            print("타원이 발견되지 않았습니다.")


    def release_resources(self):
        # 자원 해제
        cv2.destroyAllWindows()

if __name__ == "__main__":
    a = Star("cs9.png")  # 이미지 파일 경로로 변경

    a.preprocess_frame()
    edges = a.ray_casting(max_radius=600,thresh=15)
    best_ellipse = a.fit_ellipse_ransac(edges,num_iterations=500, threshold=3,min_density_threshold=1,min_length=150,max_length=400)
    a.draw_ellipse(best_ellipse, edges)
    
    cv2.imshow("Processed Image", a.frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    a.release_resources()

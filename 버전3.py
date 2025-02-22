import numpy as np
import cv2
import math

'''
only starburst + pic ver
 
< 값 변경 가능 변수 >
1) line 39 : _, self.threshold = cv2.threshold(self.gray, 50, 255, cv2.THRESH_BINARY) 에서 '50' 
● 픽셀값이 50이상이면, 255(흰색)으로 설정
● 픽셀값이 50이하면, 0(검정)으로 설정
이진화 임계값 조정가능 - 영상이 어두울 수록 값을 낮춰야함 
2) line 41 &154 : max_radius=100 영상,눈크기에 따라 값 조정해야함
3) line 67 : max_iterations=300 
'''

class Star:
    def __init__(self, image_path):
        self.image_path = image_path
        self.frame = cv2.imread(image_path)
        if self.frame is None:
            raise ValueError("Failed to load the image file.")
        
        self.image_width = self.frame.shape[1]
        self.image_height = self.frame.shape[0]

    def preprocess_frame(self):
        bright_frame = cv2.convertScaleAbs(self.frame, alpha=1.5, beta=0)
        self.gray = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2GRAY)
        return self.frame

    def ray_casting(self, num_rays=360, max_radius=450):
        self.xc, self.yc = self.image_width // 2, self.image_height // 2
        edges = []
        _, self.threshold = cv2.threshold(self.gray, 20, 255, cv2.THRESH_BINARY)
        self.sobel_x = cv2.Sobel(self.threshold, cv2.CV_64F, 1, 0, ksize=3)
        self.sobel_y = cv2.Sobel(self.threshold, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(self.sobel_x**2 + self.sobel_y**2)

        for i in range(num_rays):
            theta = 2 * np.pi * i / num_rays

            for r in range(0, max_radius):
                x = int(self.xc + r * np.cos(theta))
                y = int(self.yc + r * np.sin(theta))
                #cv2.line(self.frame, (self.xc, self.yc), (x, y), (0, 255, 0), 1) 가상 광선 길이 확인 디버깅용
                if x < 0 or x >= self.frame.shape[1] or y < 0 or y >= self.frame.shape[0]:
                    break
                sobel_value = sobel_mag[int(y), int(x)] if (0 <= x < sobel_mag.shape[1] and 0 <= y < sobel_mag.shape[0]) else 0
                if sobel_value > 70:                
                    edges.append((x, y))
                    #print('sobel_value:',sobel_value)  소벨 연산자 값 디버깅용
                    
        for x, y in edges:
            cv2.circle(self.frame, (x, y), 1, (0, 255, 0), -1)
        print('total edges:',len(edges),'개')
        
        return edges



    def ransac_apply(self, points, sample_size=5, max_iterations=500,inlier_threshold=3, stop_inlier_ratio=0.6):
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
                    
                    if ellipse[1][0] < 180 or ellipse[1][1] < 180:
                        print('ellipse too small')
                        continue
                    if  ellipse[1][0] > 300 or ellipse[1][1] > 300:
                        print('ellipse too big')
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
        print('max_inliers is:',max_inliers, ', inlier_ratio is:',inlier_ratio)
        return best_ellipse

    def draw_ellipse(self, ellipse):
        if ellipse is not None:
            center, axes, angle = ellipse
            print(f"타원의 중심: {center}, 주축,부축 길이: {axes}, 회전 각도: {angle}")
            # 타원의 넓이 계산
            a = axes[0] / 2  # 반장축 (semi-major axis)
            b = axes[1] / 2  # 반단축 (semi-minor axis)
            area = np.pi * a * b
            print(f"타원의 넓이: {area:.2f}")
            C = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
            print(f"타원 경계의 길이: {C:.2f}")
            cv2.ellipse(self.frame, ellipse, (0, 0, 255), 2)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.circle(self.frame, (int(ellipse[0][0]), int(ellipse[0][1])), 2, (255, 0, 255), -1)
            print(f"Ellipse center: {center}")
        else:
            print("No ellipse detected.")
        



if __name__ == "__main__":
    a = Star("cs7.png")  
    a.preprocess_frame()
    edges = a.ray_casting()
    best_ellipse = a.ransac_apply(edges)
    a.draw_ellipse(best_ellipse)

    cv2.imshow("Processed Image", a.frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    


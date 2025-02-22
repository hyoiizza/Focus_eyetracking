import cv2
import numpy as np

class StarRANSACOGD:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        self.frame = None
        self.edges = []

    def detect_edges(self, threshold=50):
        """스타버스트 알고리즘으로 경계점을 검출"""
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold, threshold * 2)
        y_indices, x_indices = np.where(edges > 0)
        self.edges = list(zip(x_indices, y_indices))

    def fit_ellipse_ransac(self):
        """RANSAC으로 타원 초기 피팅"""
        if len(self.edges) < 5:
            return None
        points = np.array(self.edges, dtype=np.int32)
        ellipse = cv2.fitEllipse(points)
        if ellipse[1][0] < 20 or ellipse[1][1] < 20:
                print('ellipse too small')
                ellipse = cv2.fitEllipse(points)
        if  ellipse[1][0] > 40 or ellipse[1][1] > 40:
                print('ellipse too big')
                ellipse = cv2.fitEllipse(points)
            
        return ellipse

    def optimize_with_OGD(self, ellipse, learning_rate=0.1, max_iterations=100):
        """OGD로 타원을 최적화"""
        center, axes, angle = ellipse
        cx, cy = center
        a, b = axes[0] / 2, axes[1] / 2

        for _ in range(max_iterations):
            gradient_cx, gradient_cy = 0, 0
            gradient_a, gradient_b = 0, 0

            for x, y in self.edges:
                dx = (x - cx) * np.cos(np.radians(angle)) + (y - cy) * np.sin(np.radians(angle))
                dy = -(x - cx) * np.sin(np.radians(angle)) + (y - cy) * np.cos(np.radians(angle))
                distance = (dx**2) / (a**2) + (dy**2) / (b**2)

                error = abs(distance - 1)
                if error < 0.1:
                    continue

                gradient_cx += (2 * dx / a**2) * (distance - 1)
                gradient_cy += (2 * dy / b**2) * (distance - 1)
                gradient_a += -(2 * dx**2 / a**3) * (distance - 1)
                gradient_b += -(2 * dy**2 / b**3) * (distance - 1)

            cx -= learning_rate * gradient_cx
            cy -= learning_rate * gradient_cy
            a -= learning_rate * gradient_a
            b -= learning_rate * gradient_b

            a = max(a, 1)
            b = max(b, 1)

        optimized_ellipse = ((cx, cy), (2 * a, 2 * b), angle)
        return optimized_ellipse

    def draw_ellipse(self, ellipse):
        """OGD로 최적화된 타원 시각화"""
        optimized_ellipse = self.optimize_with_OGD(ellipse)
        if optimized_ellipse is not None:
            cv2.ellipse(self.frame, optimized_ellipse, (0, 255, 0), 2)
        else:
            print("최적화된 타원을 그릴 수 없습니다.")

    def process_frame(self):
        """프레임을 처리하고 결과를 표시"""
        ret, self.frame = self.cap.read()
        if not ret:
            return False

        self.detect_edges()
        ellipse = self.fit_ellipse_ransac()
        if ellipse is not None:
            self.draw_ellipse(ellipse)

        cv2.imshow("Pupil Detection", self.frame)
        return True

    def run(self):
        """비디오 스트림 실행"""
        while True:
            if not self.process_frame():
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = StarRANSACOGD('Time2.mov')
    tracker.run()

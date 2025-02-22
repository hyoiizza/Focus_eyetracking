import matplotlib.pyplot as plt

# 타원의 중심 좌표
centers = [
    (103.94214630126953, 93.81991577148438),
    (129.51364135742188, 98.3182601928711),
    (103.19149017333984, 86.53191375732422),
    (125.43757629394531, 93.71195983886719),
    (118.98605346679688, 95.81590270996094)
]

# x, y 좌표 분리
x_coords, y_coords = zip(*centers)




# 좌표 범위를 더 작게 설정하여 확대
x_min, x_max = min(x_coords) - 5, max(x_coords) + 5
y_min, y_max = min(y_coords) - 1, max(y_coords) + 1

# 시각화 (더 큰 확대)
plt.figure(figsize=(6, 6))
plt.scatter(x_coords, y_coords, color='blue', label='Ellipse Centers', zorder=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title('동공의 시선추적점')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 각 좌표에 라벨 추가
for i, (x, y) in enumerate(centers):
    plt.text(x, y, f'P{i+1}', fontsize=9, ha='right', zorder=3)

plt.axhline(0, color='black', linewidth=0.8, zorder=1)
plt.axvline(0, color='black', linewidth=0.8, zorder=1)
plt.legend()
plt.show()

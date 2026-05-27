import numpy as np
import matplotlib.pyplot as plt

def check_points(x, y, z) -> None:
    """Проверка того что все точки на сфере и расчет среднего отклонения от радиуса"""
    r = np.sqrt(x * x + y * y + z * z)
    print(f"Среднее отклонение радиуса от 1: {np.mean(np.abs(r - 1))}")

def draw_sphere(points, n: int = 10000) -> None:
    """Визуализация сферы и точек"""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    sample = points[:n]
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1)
    ax.set_title(f"Визуализация сферы и точек ({n} точек)")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

def draw_histogram(z) -> None:
    """Проверка равномерности распределения
    В идеальном случае гистограмма Z должна быть ровной линией
    """
    plt.figure(figsize=(8, 4))
    plt.hist(z, bins=50, density=True, color='b')
    plt.title("Гистограмма распределения z = cos(theta)")
    plt.xlabel("z")
    plt.ylabel("Плотность")
    plt.grid(True)

def main() -> None:
    n = 100000
    # cos(theta) равномерно в [-1, 1]
    u = np.random.uniform(-1, 1, n)
    # phi равномерно в [0, 2π]
    phi = np.random.uniform(0, 2*np.pi, n)
    theta = np.arccos(u)
    # декартовы координаты (точки на сфере)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    # shape points = (100000, 3)
    points = np.vstack((x, y, z)).T
    check_points(x, y, z)
    draw_sphere(points)
    draw_histogram(z)
    plt.show()

if __name__ == '__main__':
    main()

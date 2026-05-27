import numpy as np
import matplotlib.pyplot as plt

def generate_sample_triangle(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, n: int) -> np.ndarray:
    """
    Функция генерации равномерных точек внутри треугольника
    Метод: u, v — независимые равномерные, но если u+v>1 – отражаем
    """
    u = np.random.rand(n)
    v = np.random.rand(n)
    # отражение для равномерности
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    # генерация точек
    return v1 + u[:,None] * (v2 - v1) + v[:,None] * (v3 - v1)

def visualization_dots(triangle_projection, projection_x, projection_y, n: int = 10000) -> None:
    """Визуализация треугольника с отмеченными n точек"""
    plt.figure(figsize=(5, 5))
    plt.fill(
        triangle_projection[:, 0],
        triangle_projection[:, 1],
        edgecolor='black',
        fill=False,
        linewidth=2
    )
    plt.scatter(projection_x[:n], projection_y[:n], s=1, color='blue')
    plt.title(f"Треугольник и первые {n} точек")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def visualization_of_distribution_heat_map(projection_x, projection_y) -> None:
    """Визуализация тепловой карты распределения"""
    plt.figure(figsize=(5, 5))
    plt.hist2d(projection_x, projection_y, bins=500, density=True)
    plt.title("Тепловая карта плотности точек")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Плотность")
    plt.show()


def main() -> None:
    # число точек
    n = 1000000
    # координаты треугольника
    v1 = np.array([1, 2, 0])
    v2 = np.array([3, 5, 2])
    v3 = np.array([0, 4, 3])
    points = generate_sample_triangle(v1, v2, v3, n)
    # Чтобы визуализировать распределение, надо перевести точки в 2D (систему координат треугольника)
    e1 = v2 - v1
    e2 = v3 - v1
    # ортонормируем базис
    e1n = e1 / np.linalg.norm(e1)
    e2p = e2 - np.dot(e2, e1n) * e1n
    e2n = e2p / np.linalg.norm(e2p)
    # проекция на 2d
    projection_x = np.dot(points - v1, e1n)
    projection_y = np.dot(points - v1, e2n)
    triangle_projection = np.array([
        [np.dot(v1 - v1, e1n), np.dot(v1 - v1, e2n)],
        [np.dot(v2 - v1, e1n), np.dot(v2 - v1, e2n)],
        [np.dot(v3 - v1, e1n), np.dot(v3 - v1, e2n)],
    ])
    # визуализация
    visualization_dots(triangle_projection, projection_x, projection_y)
    visualization_of_distribution_heat_map(projection_x, projection_y)

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Вектор нулевой длины")
    return v / n

def generate_points(C, Rc, number_points, u_axis, v_axis):
    """Генерируем равномерно распределенные точки внутри диска"""
    u = np.random.rand(number_points)
    theta = 2 * np.pi * np.random.rand(number_points)
    r = Rc * np.sqrt(u)
    x_local = r * np.cos(theta)
    y_local = r * np.sin(theta)
    return C[None, :] + np.outer(x_local, u_axis) + np.outer(y_local, v_axis)

def check_uniformity(Rc, radial, counts, density) -> None:
    """Проверка статистической равномерности"""
    print("Результат проверки статистической равномерности:")
    print(f"min:{counts.min()}, max: {counts.max()}, mean: {counts.mean()}")
    print("статистика плотностей:")
    print(f"средняя плотность = {density.mean()}")
    print(f"стандартное отклонение = {density.std()}")
    # Вывод результатов
    print("Контрольные проверки:")
    print(f"Максимальный радиус меньше или равен Rc: {radial.max() <= Rc + 1e-12}")
    cv = density.std() / density.mean()
    print(f"Коэффициент вариации плотности: {cv}")
    print("CV < 0.1 обычно приемлемо для равномерности" if cv < 0.1 else "CV слишком велик")

def draw_first_figure(
    C,
    Rc,
    number_points,
    points,
    u_axis,
    v_axis,
    x_projection,
    y_projection,
    number_points_for_draw=10000
) -> None:
    """Визуализация генерации точек в плоскости круга"""
    subset_idx = np.random.choice(number_points, size=number_points_for_draw, replace=False)
    pts3 = points[subset_idx]
    # Подмножество точек и окружность в 3D
    first_figure = plt.figure(figsize=(12, 6))
    graph_first = first_figure.add_subplot(1, 2, 1, projection='3d')
    graph_first.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2], s=1)
    angles = np.linspace(0, 2 * np.pi, 200)
    rim = C[None, :] + np.outer(Rc * np.cos(angles), u_axis) + np.outer(Rc * np.sin(angles), v_axis)
    graph_first.plot(rim[:, 0], rim[:, 1], rim[:, 2], 'b', linewidth=1.2)
    graph_first.set_title("Подмножество точек и окружность в 3D")
    graph_first.set_xlabel('X')
    graph_first.set_ylabel('Y')
    graph_first.set_zlabel('Z')
    # Проекция на плоскость окружности
    graph_second = first_figure.add_subplot(1, 2, 2)
    graph_second.scatter(x_projection[subset_idx], y_projection[subset_idx], s=1)
    circle = plt.Circle((0, 0), Rc, edgecolor='b', facecolor='none')
    graph_second.add_patch(circle)
    graph_second.set_xlim(-Rc * 1.1, Rc * 1.1)
    graph_second.set_ylim(-Rc * 1.1, Rc * 1.1)
    graph_second.set_title("Проекция на плоскость окружности")
    plt.tight_layout()

def draw_second_figure(Rc, radial) -> None:
    """Распределение плотностей генерируемых точек"""
    second_figure = plt.figure(figsize=(12, 6))
    # pdf(r) - функция плотности вероятности
    graph_third = second_figure.add_subplot(1, 2, 1)
    hist_counts, hist_bins = np.histogram(radial, bins=100, density=True)
    bin_centers_second = 0.5 * (hist_bins[:-1] + hist_bins[1:])
    theo = 2 * bin_centers_second / (Rc ** 2)
    graph_third.plot(bin_centers_second, hist_counts, label='эмпирическая pdf(r)')
    graph_third.plot(bin_centers_second, theo, '--', label='теоретическая pdf=2r/Rc^2')
    graph_third.set_xlabel('r')
    graph_third.set_ylabel('pdf')
    graph_third.set_title('Радиальное распределение')
    graph_third.legend()
    graph_third.grid(True)

def main() -> None:
    # Исходные параметры
    C = np.array([1.0, 1.5, 2.0]) # центр окружности в 3D
    vector_n = np.array([0.2, -1.0, 0.5]) # вектор нормали (ориентация)
    Rc = 1.5 # радиус
    number_points = 100000 # количество точек
    # Рассчитываем ортогональный базис (u_axis, v_axis) для плоскости круга
    Nn = normalize(vector_n)
    if abs(Nn[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])
    u_axis = normalize(np.cross(Nn, tmp))
    v_axis = normalize(np.cross(Nn, u_axis))
    points = generate_points(C, Rc, number_points, u_axis, v_axis)
    # Проверка расстояний
    rel = points - C[None, :]
    x_projection = np.dot(rel, u_axis)
    y_projection = np.dot(rel, v_axis)
    radial = np.sqrt(x_projection**2 + y_projection**2)
    n_bins = 50
    bins = np.linspace(0, Rc, n_bins + 1)
    counts, _ = np.histogram(radial, bins=bins)
    areas = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    density = counts / areas
    print("Количество точек:", number_points)
    check_uniformity(Rc, radial, counts, density)
    draw_first_figure(C, Rc, number_points, points, u_axis, v_axis, x_projection, y_projection)
    draw_second_figure(Rc, radial)
    plt.show()

if __name__ == '__main__':
    main()

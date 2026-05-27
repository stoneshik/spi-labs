import numpy as np

def calculate_illumination(
    I0_RGB: tuple[float, float, float],
    vect_O: tuple[float, float, float],
    vect_P_L: tuple[float, float, float],
    vect_P0: tuple[float, float, float],
    vect_P1: tuple[float, float, float],
    vect_P2: tuple[float, float, float],
    x_values: tuple[float],
    y_values: tuple[float]
) -> dict[tuple[float, float], tuple[float, float, float]]:
    """
    Расчет освещенности точек на плоскости треугольника
    Параметры:
    I0_RGB - кортеж (R, G, B) интенсивности источника света
    vect_O - кортеж (x, y, z) направления оси источника (единичный вектор)
    vect_P_L - кортеж (x, y, z) положения источника света
    vect_P0, vect_P1, vect_P2 - кортежи вершин треугольника
    x_values - список значений x координат [x₁, x₂, x₃, x₄, x₅]
    y_values - список значений y координат [y₁, y₂, y₃, y₄, y₅]
    Результат:
    results - словарь полученных значений в формате (x,y): (E_RGB_x, E_RGB_y, E_RGB_z)
    """
    # Преобразование в numpy массивы
    I0: np.ndarray = np.array(I0_RGB, dtype=float)
    O: np.ndarray = np.array(vect_O, dtype=float)
    P_L: np.ndarray = np.array(vect_P_L, dtype=float)
    P0: np.ndarray = np.array(vect_P0, dtype=float)
    P1: np.ndarray = np.array(vect_P1, dtype=float)
    P2: np.ndarray = np.array(vect_P2, dtype=float)
    # Проверка: vect(O) должен быть единичным
    O_norm: np.floating = np.linalg.norm(O)
    if abs(O_norm - 1.0) > 1e-10:
        O = O / O_norm
        print(f"Предупреждение: vect(O) нормализован до единичной длины")
    # Вычисление нормали плоскости
    v1: np.ndarray = P1 - P0
    v2: np.ndarray = P2 - P0
    N: np.ndarray = np.cross(v1, v2)  # Внешняя нормаль
    N_norm: np.floating = np.linalg.norm(N)
    if N_norm == 0:
        raise ValueError("Точки треугольника коллинеарны, нормаль не определена")
    N = N / N_norm  # Нормализация нормали
    # Единичные векторы ребер для локальной системы координат
    u1: np.ndarray = v1 / np.linalg.norm(v1)
    u2: np.ndarray = v2 / np.linalg.norm(v2)
    # генерируем список всех точек
    points: list[tuple[float, float]] = [(x, y) for x in x_values for y in y_values]
    # Начинаем вычислять значения освещенности для каждой точки
    results: dict[tuple[float, float], tuple[float, float, float]] = {}
    for x, y in points:
        try:
            # 1. Перевод локальных координат в глобальные
            P_T: np.ndarray = P0 + u1 * x + u2 * y
            # 2. Вектор от точки к источнику света и расстояние
            s: np.ndarray = P_L - P_T
            R: np.floating = np.linalg.norm(s)
            if R < 1e-10:  # Избегаем деления на ноль
                results[(x, y)] = (0.0, 0.0, 0.0)
                continue
            # 3. Расчет интенсивности I(RGB, vect(s))
            cos_theta: float = np.dot(s, O) / R
            I_RGB: np.ndarray = I0 * abs(cos_theta)
            # 4. Расчет освещенности E(RGB, vect(P_T))
            cos_alpha: float = np.dot(s, N) / R
            E_RGB: np.ndarray = I_RGB * abs(cos_alpha) / (R * R)
            E_RGB_x, E_RGB_y, E_RGB_z = E_RGB
            results[(x, y)] = (E_RGB_x, E_RGB_y, E_RGB_z)
        except Exception as e:
            print(f"Ошибка при расчете точки ({x}, {y}): {e}")
            results[(x, y)] = (0.0, 0.0, 0.0)
    return results

def print_results_table(
    results: dict[tuple[float, float], tuple[float, float, float]],
    x_values: tuple[float],
    y_values: tuple[float]
) -> None:
    """Красивое отображение результатов в виде таблицы"""
    print("Освещенность E(RGB, vect(P)) для точек треугольника:")
    # Определяем ширину столбцов
    num_cols = len(x_values)
    col_width = 17
    # Верхняя граница таблицы
    top_border = "+" + "-" * 6 + "+" + ("-" * (col_width + 4) + "+") * num_cols
    print(top_border)
    # Заголовок с x координатами
    header = "| y\\x  |"
    for x in x_values:
        header += f" {x:^{col_width + 2}.2f} |"
    print(header)
    print(top_border)
    # Отображение данных
    for y in y_values:
        row = f"| {y:3.2f} |"
        for x in x_values:
            E = results.get((x, y), (0, 0, 0))
            # Форматируем RGB значения
            rgb_str = f"{E[0]:.3f}, {E[1]:.3f}, {E[2]:.3f}"
            row += f" {rgb_str:^{col_width - 1}} |"
        print(row)
        print(top_border)

def main() -> None:
    # Входные данные
    I0_RGB = (50.0, 20.8, 15.6)  # I₀(RGB)
    vect_O = (0, -0.2, 1)  # Ось источника
    vect_P_L = (0, 0, 5)  # Положение источника
    vect_P0 = (0, 0, 0)  # Вершины треугольника
    vect_P1 = (2, 0, 0)
    vect_P2 = (0, 2, 0)
    # Значения x₁-x₅ и y₁-y₅
    x_1, x_2, x_3, x_4, x_5 = 0.0, 0.75, 1.5, 2.25, 3.0
    y_1, y_2, y_3, y_4, y_5 = 0.0, 0.3, 0.5, 1.0, 1.5
    x_values: tuple = (x_1, x_2, x_3, x_4, x_5)
    y_values: tuple = (y_1, y_2, y_3, y_4, y_5)
    # Вывод входных параметров
    print("Входные параметры:")
    print(f"x₁-x₅: {x_values}")
    print(f"y₁-y₅: {y_values}")
    print(f"I₀(RGB): {I0_RGB}")
    print(f"vect(O): {vect_O}")
    print(f"vect(P_L): {vect_P_L}")
    print(f"Треугольник: P₀={vect_P0}, P₁={vect_P1}, P₂={vect_P2}")
    print()
    # Расчет освещенности
    results = calculate_illumination(I0_RGB, vect_O, vect_P_L, vect_P0, vect_P1, vect_P2, x_values, y_values)
    # Вывод результатов
    print_results_table(results, x_values, y_values)
    print(f"\nДополнительная информация для проверки расчетов:")
    v1 = np.array(vect_P1) - np.array(vect_P0)
    v2 = np.array(vect_P2) - np.array(vect_P0)
    N = np.cross(v1, v2)
    N_norm = N / np.linalg.norm(N)
    print(f"Нормаль плоскости: ({N_norm[0]:.3f}, {N_norm[1]:.3f}, {N_norm[2]:.3f})")

if __name__ == "__main__":
    main()

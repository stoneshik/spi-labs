import numpy as np
import math

def calculate_brightness(
    lights: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    vect_P0: tuple[float, float, float],
    vect_P1: tuple[float, float, float],
    vect_P2: tuple[float, float, float],
    x_values: tuple[float, ...],
    y_values: tuple[float, ...],
    vect_v: tuple[float, float, float],
    K_RGB: tuple[float, float, float],
    k_d: float,
    k_s: float,
    k_e: float
) -> dict[tuple[float, float], tuple[float, float, float]]:
    """
    Расчет яркости точек на плоскости треугольника с учетом освещения и BRDF
    Параметры:
    lights - список источников света, каждый источник: (I0_RGB, vect_O, vect_P_L)
    vect_P0, vect_P1, vect_P2 - вершины треугольника
    x_values, y_values - локальные координаты точек
    vect_v - направление наблюдения (единичный вектор)
    K_RGB - цвет поверхности (R, G, B)
    k_d - коэффициент диффузного отражения
    k_s - коэффициент зеркальности
    k_e - коэффициент ширины блика
    Результат:
    results - словарь: (x, y): (L_R, L_G, L_B)
    """
    # Преобразование в numpy массивы
    P0 = np.array(vect_P0, dtype=float)
    P1 = np.array(vect_P1, dtype=float)
    P2 = np.array(vect_P2, dtype=float)
    v = np.array(vect_v, dtype=float)
    K = np.array(K_RGB, dtype=float)
    # Нормализация вектора наблюдения
    v_norm = np.linalg.norm(v)
    if abs(v_norm - 1.0) > 1e-10:
        v = v / v_norm
    # Вычисление нормали плоскости
    v1 = P1 - P0
    v2 = P2 - P0
    N = np.cross(v1, v2)
    N_norm = np.linalg.norm(N)
    if N_norm == 0:
        raise ValueError("Точки треугольника коллинеарны, нормаль не определена")
    N = N / N_norm
    # Если скалярное произведение вектора наблюдения и нормали отрицательное,
    # значит наблюдатель находится с противоположной стороны от нормали
    if np.dot(v, N) < 0:
        N = -N  # Разворачиваем нормаль к наблюдателю
    # Единичные векторы ребер для локальной системы координат
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    # Подготовка источников света
    processed_lights = []
    for I0_RGB, vect_O, vect_P_L in lights:
        I0 = np.array(I0_RGB, dtype=float)
        O = np.array(vect_O, dtype=float)
        P_L = np.array(vect_P_L, dtype=float)
        # Нормализация оси источника
        O_norm = np.linalg.norm(O)
        if abs(O_norm - 1.0) > 1e-10:
            O = O / O_norm
        processed_lights.append((I0, O, P_L))
    # Генерация списка всех точек
    points = [(x, y) for x in x_values for y in y_values]
    # Расчет яркости для каждой точки
    results = {}
    for x, y in points:
        try:
            # 1. Перевод локальных координат в глобальные
            P_T = P0 + u1 * x + u2 * y
            # 2. Расчет вклада от всех источников света
            total_contribution = np.zeros(3)
            for I0, O, P_L in processed_lights:
                # Вектор от точки к источнику света
                s = P_L - P_T
                R = np.linalg.norm(s)
                if R < 1e-10:  # Избегаем деления на ноль
                    continue
                # Нормализованный вектор к источнику
                s_norm = s / R
                # Расчет интенсивности I(RGB, vect(s))
                cos_theta = np.dot(s_norm, O)
                if cos_theta <= 0:  # Источник не светит в эту точку
                    continue
                I_RGB = I0 * cos_theta
                # Расчет освещенности E(RGB, vect(P_T))
                cos_alpha = np.dot(s_norm, N)
                if cos_alpha <= 0:  # Источник с противоположной стороны или в плоскости
                    continue
                E_RGB = I_RGB * cos_alpha / (R * R)
                # Расчет BRDF
                # Средний вектор
                h = v + s_norm
                h_norm = np.linalg.norm(h)
                if h_norm < 1e-10:
                    continue
                h = h / h_norm
                # Коэффициент бликового отражения
                specular = k_s * (abs(np.dot(h, N)) ** k_e)
                # BRDF функция
                f_RGB = K * (k_d + specular)
                # Вклад от данного источника
                total_contribution += E_RGB * f_RGB
            # Итоговая яркость
            L_RGB = total_contribution / math.pi
            results[(x, y)] = (L_RGB[0], L_RGB[1], L_RGB[2])
        except Exception as e:
            print(f"Ошибка при расчете точки ({x}, {y}): {e}")
            results[(x, y)] = (0.0, 0.0, 0.0)
    return results


def calculate_illumination_single_source(
    I0_RGB: tuple[float, float, float],
    vect_O: tuple[float, float, float],
    vect_P_L: tuple[float, float, float],
    vect_P0: tuple[float, float, float],
    vect_P1: tuple[float, float, float],
    vect_P2: tuple[float, float, float],
    x_values: tuple[float, ...],
    y_values: tuple[float, ...]
) -> dict[tuple[float, float], tuple[float, float, float]]:
    """
    Расчет освещенности от одного источника (из предыдущей лабораторной)
    """
    # Преобразование в numpy массивы
    I0 = np.array(I0_RGB, dtype=float)
    O = np.array(vect_O, dtype=float)
    P_L = np.array(vect_P_L, dtype=float)
    P0 = np.array(vect_P0, dtype=float)
    P1 = np.array(vect_P1, dtype=float)
    P2 = np.array(vect_P2, dtype=float)
    # Нормализация оси источника
    O_norm = np.linalg.norm(O)
    if abs(O_norm - 1.0) > 1e-10:
        O = O / O_norm
    # Вычисление нормали плоскости
    v1 = P1 - P0
    v2 = P2 - P0
    N = np.cross(v1, v2)
    N_norm = np.linalg.norm(N)
    if N_norm == 0:
        raise ValueError("Точки треугольника коллинеарны, нормаль не определена")
    N = N / N_norm
    # Единичные векторы ребер
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    points = [(x, y) for x in x_values for y in y_values]
    results = {}
    for x, y in points:
        try:
            # Перевод локальных координат в глобальные
            P_T = P0 + u1 * x + u2 * y
            # Вектор к источнику света
            s = P_L - P_T
            R = np.linalg.norm(s)
            if R < 1e-10:
                results[(x, y)] = (0.0, 0.0, 0.0)
                continue
            # Расчет интенсивности
            cos_theta = np.dot(s, O) / R
            I_RGB = I0 * abs(cos_theta)
            # Расчет освещенности
            cos_alpha = np.dot(s, N) / R
            E_RGB = I_RGB * abs(cos_alpha) / (R * R)
            results[(x, y)] = (E_RGB[0], E_RGB[1], E_RGB[2])
        except Exception as e:
            print(f"Ошибка при расчете точки ({x}, {y}): {e}")
            results[(x, y)] = (0.0, 0.0, 0.0)
    return results


def print_results_table(
    results: dict[tuple[float, float], tuple[float, float, float]],
    x_values: tuple[float, ...],
    y_values: tuple[float, ...],
    title: str = "Результаты"
) -> None:
    """Красивое отображение результатов в виде таблицы"""
    print(f"{title}:")
    num_cols = len(x_values)
    col_width = 19
    # Верхняя граница таблицы
    top_border = "+" + "-" * 9 + "+" + ("-" * (col_width + 2) + "+") * num_cols
    print(top_border)
    # Заголовок с x координатами
    header = "|   y\\x   |"
    for x in x_values:
        header += f" {x:^{col_width}.2f} |"
    print(header)
    print(top_border)
    # Отображение данных
    for y in y_values:
        row = f"| {y:>6.2f}  |"
        for x in x_values:
            values = results.get((x, y), (0, 0, 0))
            # Форматируем RGB значения
            rgb_str = f"{values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f}"
            row += f" {rgb_str:^{col_width}} |"
        print(row)
        print(top_border)

def main() -> None:
    # Входные данные для двух источников света
    # Источник 1
    I0_1_RGB = (80.0, 60.0, 40.0)  # Более теплый свет
    vect_O_1 = (0.1, -0.2, 1.0)  # Ось первого источника
    vect_P_L_1 = (1.0, 1.0, 4.0)  # Положение первого источника
    # Источник 2  
    I0_2_RGB = (40.0, 60.0, 80.0)  # Более холодный свет
    vect_O_2 = (-0.1, -0.3, 1.0)  # Ось второго источника
    vect_P_L_2 = (-1.0, 2.0, 3.0)  # Положение второго источника
    # Геометрия треугольника
    vect_P0 = (0.0, 0.0, 0.0)
    vect_P1 = (2.0, 0.0, 0.0)
    vect_P2 = (0.0, 2.0, 0.0)
    # Сетка точек
    x_1, x_2, x_3, x_4, x_5 = 0.0, 0.5, 1.0, 1.5, 2.0
    y_1, y_2, y_3, y_4, y_5 = 0.0, 0.4, 0.8, 1.2, 1.6
    x_values = (x_1, x_2, x_3, x_4, x_5)
    y_values = (y_1, y_2, y_3, y_4, y_5)
    # Параметры наблюдения и материала
    vect_v = (0.0, 0.0, 1.0)  # Направление наблюдения (сверху)
    K_RGB = (0.8, 0.6, 0.4)  # Цвет поверхности (коричневатый)
    k_d = 0.7  # Коэффициент диффузного отражения
    k_s = 0.3  # Коэффициент зеркальности
    k_e = 8.0  # Коэффициент ширины блика
    # Вывод входных параметров
    print("Входные параметры:")
    print(f"1. I_O_1(RGB) = {I0_1_RGB}; I_O_2(RGB) = {I0_2_RGB}")
    print(f"2. vect(O₁) = {vect_O_1}; vect(O₂) = {vect_O_2}")
    print(f"3. vect(P_L_1) = {vect_P_L_1}; vect(P_L_2) = {vect_P_L_2}")
    print(f"4. vect(P₀) = {vect_P0}")
    print(f"5. vect(P₁) = {vect_P1}")
    print(f"6. vect(P₂) = {vect_P2}")
    print(f"7. x₁={x_1}, y₁={y_1}; x₂={x_2}, y₂={y_2}; x₃={x_3}, y₃={y_3}; x₄={x_4}, y₄={y_4}; x₅={x_5}, y₅={y_5}")
    print(f"8. vect(V) = {vect_v}")
    print(f"9. K(RGB) = {K_RGB}")
    print(f"10. k_d = {k_d}")
    print(f"11. k_s = {k_s}")
    print(f"12. k_e = {k_e}")
    # Расчет освещенности от первого источника
    print()
    print("Расчет освещенности от первого источника:")
    results_E1 = calculate_illumination_single_source(
        I0_1_RGB, vect_O_1, vect_P_L_1, vect_P0, vect_P1, vect_P2, x_values, y_values
    )
    print_results_table(
        results_E1, x_values, y_values,"Освещенность E₁(RGB,vect(P)) от первого источника"
    )
    # Расчет освещенности от второго источника
    print()
    print("Расчет освещенности от второго источника:")
    results_E2 = calculate_illumination_single_source(
        I0_2_RGB, vect_O_2, vect_P_L_2, vect_P0, vect_P1, vect_P2, x_values, y_values
    )
    print_results_table(
        results_E2, x_values, y_values,"Освещенность E₂(RGB,vect(P)) от второго источника"
    )
    # Расчет яркости с учетом обоих источников и BRDF
    print()
    print("Расчет яркости с учетом BRDF и всех источников:")
    # Список всех источников света
    lights = [
        (I0_1_RGB, vect_O_1, vect_P_L_1),
        (I0_2_RGB, vect_O_2, vect_P_L_2)
    ]
    results_L = calculate_brightness(
        lights, vect_P0, vect_P1, vect_P2, x_values, y_values,
        vect_v, K_RGB, k_d, k_s, k_e
    )
    print_results_table(
        results_L, x_values, y_values, "Яркость L(RGB,vect(P_T),vect(v))"
    )
    # Дополнительная информация
    print()
    print("Дополнительная информация для проверки:")
    # Нормаль плоскости
    v1 = np.array(vect_P1) - np.array(vect_P0)
    v2 = np.array(vect_P2) - np.array(vect_P0)
    N = np.cross(v1, v2)
    N_norm = N / np.linalg.norm(N)
    print(f"Нормаль плоскости: ({N_norm[0]:.3f}, {N_norm[1]:.3f}, {N_norm[2]:.3f})")
    # Проверка входных векторов
    O1_norm = np.linalg.norm(vect_O_1)
    O2_norm = np.linalg.norm(vect_O_2)
    v_norm = np.linalg.norm(vect_v)
    print(f"Длина vect(O₁): {O1_norm:.3f}, vect(O₂): {O2_norm:.3f}, vect(v): {v_norm:.3f}")

if __name__ == "__main__":
    main()

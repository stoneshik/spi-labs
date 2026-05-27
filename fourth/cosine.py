import numpy as np
import matplotlib.pyplot as plt

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Вектор нулевой длины")
    return v / n

def check(R, O, cos_alpha, P) -> None:
    # Проверка, что P действительно на сфере радиуса R вокруг O
    distances_to_center = np.linalg.norm(P - O, axis=1)
    print(f"Расстояния до центра: {distances_to_center.min():.6f} (min), {distances_to_center.max():.6f} (max)")
    print(f"Все точки на сфере: {np.allclose(distances_to_center, R, atol=1e-10)}")
    # cos(α) должен иметь распределение p(c) = 2c на [0, 1]
    print(f"cos(α): {cos_alpha.min():.6f} (min), {cos_alpha.max():.6f} (max)")
    print(f"E[cos(α)] = {cos_alpha.mean():.6f} (теоретическое: {2 / 3:.6f})")

def draw_first_figure(C, R, Nn, O, cos_alpha, P, n, number_points_for_draw=2000) -> None:
    subset = np.random.choice(n, size=number_points_for_draw, replace=False)
    figure = plt.figure(figsize=(13, 5))
    # точки P на сфере
    graph_first = figure.add_subplot(1, 2, 1, projection="3d")
    graph_first.scatter(P[subset, 0], P[subset, 1], P[subset, 2], s=1)
    graph_first.scatter(C[0], C[1], C[2], s=20, color="g", label="C")
    graph_first.scatter(O[0], O[1], O[2], s=20, color="r", label="O (центр)")
    graph_first.quiver(C[0], C[1], C[2], Nn[0], Nn[1], Nn[2], length=R, color="black", label="N ось")
    graph_first.set_title("Точки P на сфере")
    graph_first.set_box_aspect([1, 1, 1])
    graph_first.legend()
    # Гистограмма cos(alpha) и теоретическая плотность 2c
    graph_second = figure.add_subplot(1, 2, 2)
    bins = 50
    graph_second.hist(cos_alpha, bins=bins, range=(0, 1), density=True, alpha=0.7, label="эмпирические")
    c_grid = np.linspace(0, 1, 500)
    graph_second.plot(c_grid, 2 * c_grid, linewidth=3, color="b", label="теор: f(c) = 2c")
    graph_second.set_title("cos(alpha) = U*N")
    graph_second.set_xlabel("c = cos(alpha)")
    graph_second.set_ylabel("плотность")
    graph_second.grid(True)
    graph_second.legend()
    plt.tight_layout()
    plt.show()

def main() -> None:
    n = 100000
    C = np.array([0.5, -0.2, 0.3], dtype=float)
    vector_n = np.array([0.2, 0.6, 0.76], dtype=float)  # ось N
    R = 1.0
    Nn = normalize(vector_n)
    # ортонормальный базис вокруг N
    # строим два вектора T, B такие, что {T, B, Nn} — ортонормальный базис
    if abs(Nn[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0])
    else:
        a = np.array([1.0, 0.0, 0.0])
    T = normalize(np.cross(a, Nn))
    B = np.cross(Nn, T) # уже нормированный из-за ортонормальности
    # p(ω) = (cos α)/π на полусфере u*N >= 0
    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.sqrt(1.0 - u1)
    # локальный вектор (x, y, z) -> мировой u = x*T + y*B + z*Nn
    U = (
        x[:, None] * T[None, :] +
        y[:, None] * B[None, :] +
        z[:, None] * Nn[None, :]
    )
    # дополнительно нормируем
    U = U / np.linalg.norm(U, axis=1)[:, None]
    # строим точки P на сфере, чтобы |P-C| ∝ cos(alpha)
    # центр сферы O = C + R * Nn (C - "южный полюс" сферы)
    O = C + R * Nn
    cos_alpha = U @ Nn # должно быть в [0,1]
    t = 2.0 * R * cos_alpha # это |P - C| (длина хорды), t ∈ [0, 2R]
    P = C + t[:, None] * U # пересечение луча со сферой (кроме t=0)
    # Проверка и графики
    check(R, O, cos_alpha, P)
    draw_first_figure(C, R, Nn, O, cos_alpha, P, n)

if __name__ == "__main__":
    main()

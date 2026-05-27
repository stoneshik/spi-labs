import numpy as np
from numpy import floating
from typing import Any

def f(x) -> floating[Any]:
    """Целевая функция"""
    return x ** 2

def analytical_integral() -> float:
    """Аналитическое вычисление интеграла"""
    return 39.0

class MonteCarlo:
    def __init__(self, a=2, b=5):
        self.a = a
        self.b = b
        self.true_value = analytical_integral()
        self.n_samples_list = [100, 1000, 10000, 100000]

    def mc_simple_uniform(self, N: int) -> floating[Any]:
        """
        Простое интегрирование методом Монте-Карло:
        X ~ U(a, b)
        I ≈ (b-a)/N * sum f(X_i)
        """
        x = np.random.uniform(self.a, self.b, size=N)
        return (self.b - self.a) * np.mean(f(x))

    def mc_stratified(self, N: int, step: float) -> float:
        """
        Стратификация по отрезкам длины step:
        [a, a+step], [a+step, a+2*step], ..., [.., b]
        В каждом страта-отрезке берём примерно одинаковое число выборок.
        """
        edges = np.arange(self.a, self.b + 1e-12, step)
        if abs(edges[-1] - self.b) > 1e-9:
            edges = np.append(edges, self.b)
        total_estimate = 0.0
        n_strata = len(edges) - 1
        # равномерно распределяем N по стратам
        base = N // n_strata
        remainder = N - base * n_strata
        for i in range(n_strata):
            n_i = base + (1 if i < remainder else 0)
            left, right = edges[i], edges[i + 1]
            x = np.random.uniform(left, right, size=n_i)
            # локальный интеграл
            total_estimate += (right - left) * np.mean(f(x))
        return total_estimate

    """
    ====================================================================
    Вспомогательные функции для выборки по значимости
    p(x) ~ x^k,
    k = 1,2,3 на [a, b]
    ====================================================================
    """
    def sample_power(self, N: int, k: int) -> floating[Any]:
        """
        Сэмплирование из распределения с плотностью
        p(x)∝x^k на [a, b] через метод обратной функции.
        CDF: F(x) = (x^{k+1} - a^{k+1}) / (b^{k+1} - a^{k+1}) =>
        x = (u * (b^{k+1} - a^{k+1}) + a^{k+1})^{1/(k+1)}
        """
        u = np.random.rand(N)
        return (
            (u * (self.b ** (k + 1) - self.a ** (k + 1)) +
                self.a ** (k + 1)) ** (1.0 / (k + 1))
        )

    def pdf_power(self, x: floating[Any], k: int) -> floating[Any]:
        """
        Нормированная плотность:
        p_k(x) = (k+1) * x^k / (b^{k+1} - a^{k+1}), x∈[a,b]
        """
        return (k + 1) * x ** k / (self.b ** (k + 1) - self.a ** (k + 1))

    def mc_importance(self, N: int, k: int) -> floating[Any]:
        """
        Интегрирование методом Монте-Карло с выборкой по значимости
        для плотности p_k(x) ~ x^k (нормированной).
        I ≈ 1/N * Σ f(x_i) / p_k(x_i), где x_i ~ p_k.
        """
        x = self.sample_power(N, k)
        p = self.pdf_power(x, k)
        return np.mean(f(x) / p)
    """
    ====================================================================
    """

    def mc_mis(self, N: int, use_power_heuristic=False) -> floating[Any]:
        """
        Multiple Importance Sampling (Многократная выборка по значимости)
        p1(x) ~ x,
        p2(x) ~ x^3
        Два варианта весов:
        1) w1 = p1/(p1+p2); w2 = p2/(p1+p2)
        2) w1 = p1^2/(p1^2+p2^2); w2 = p2^2/(p1^2+p2^2)
        - N/2 выборок из p1(x) ~ x
        - N/2 выборок из p2(x) ~ x^3
        Веса считаем по заданным формулам.
        Возвращаем оценку интеграла.
        """
        N1 = N // 2
        # на случай нечётного N, но у нас все N четные
        N2 = N - N1
        # выборка из p1 (k=1)
        x1 = self.sample_power(N1, 1)
        p1_x1 = self.pdf_power(x1, 1)
        p2_x1 = self.pdf_power(x1, 3)
        # выборка из p2 (k=3)
        x2 = self.sample_power(N2, 3)
        p1_x2 = self.pdf_power(x2, 1)
        p2_x2 = self.pdf_power(x2, 3)
        if use_power_heuristic:
            # второй вариант: средний квадрат плотностей
            w1_x1 = p1_x1 ** 2 / (p1_x1 ** 2 + p2_x1 ** 2)
            w2_x2 = p2_x2 ** 2 / (p1_x2 ** 2 + p2_x2 ** 2)
        else:
            # первый вариант: средняя плотность
            w1_x1 = p1_x1 / (p1_x1 + p2_x1)
            w2_x2 = p2_x2 / (p1_x2 + p2_x2)
        # оценка из выборок p1 и p2
        estimate_first = np.mean(f(x1) * w1_x1 / p1_x1)
        estimate_second = np.mean(f(x2) * w2_x2 / p2_x2)
        # суммируем обе части
        return estimate_first + estimate_second

    def mc_russian_roulette(self, N: int, R: float) -> floating[Any]:
        """
        Метод Монте-Карло с русской рулеткой.
        Пусть с вероятностью R "выживаем" и считаем вклад,
        с вероятностью (1-R) - обнуляем вклад.
        Если выжили: X~U(a, b), вклад = (b-a)*f(X)/R
        Тогда E[вклад] = ∫f(x)dx — оценка интеграла.
        I ≈ 1/N * Σ sample_i
        """
        u = np.random.rand(N)
        survive_mask: np.ndarray[bool] = (u < R)
        M = survive_mask.sum()
        samples = np.zeros(N)
        if M > 0:
            x = np.random.uniform(self.a, self.b, size=M)
            samples[survive_mask] = (self.b - self.a) * f(x) / R
        return samples.mean()


def print_table(results: list, N: int) -> None:
    print(f"N={N}")
    header = f"{'Метод':25s} | {'I_MC' + ' ' * 8}  | {'|I_MC-I_true|'} | {'ΔI=I_true/N':>10s}  |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for name, I_est, err, delta_I in results:
        print(f"{name:25s} |  {I_est:3.8f}  |  {err:3.8f}   |  {delta_I:3.8f}  |")
    print("-" * len(header))
    print()

def main() -> None:
    # фиксируем зерно для воспроизводимости
    np.random.seed(10)
    I_true = analytical_integral()
    monte_carlo = MonteCarlo()
    print(f"Истинное значение интеграла I_true = {I_true:.6f}\n")
    for n in monte_carlo.n_samples_list:
        results = []
        delta_I = I_true / n  # оценка погрешности
        I_estimate = monte_carlo.mc_simple_uniform(n)
        results.append(("Простой", I_estimate, abs(I_estimate - I_true), delta_I))
        I_estimate = monte_carlo.mc_stratified(n, step=1.0)
        results.append(("Стратификация, шаг 1.0", I_estimate, abs(I_estimate - I_true), delta_I))
        I_estimate = monte_carlo.mc_stratified(n, step=0.5)
        results.append(("Стратификация, шаг 0.5", I_estimate, abs(I_estimate - I_true), delta_I))
        I_estimate = monte_carlo.mc_importance(n, k=1)
        results.append(("Выборка p(x)∝x", I_estimate, abs(I_estimate - I_true), delta_I))
        I_estimate = monte_carlo.mc_importance(n, k=2)
        results.append(("Выборка p(x)∝x^2", I_estimate, abs(I_estimate - I_true), delta_I))
        I_estimate = monte_carlo.mc_importance(n, k=3)
        results.append(("Выборка p(x)∝x^3", I_estimate, abs(I_estimate - I_true), delta_I))
        I_estimate = monte_carlo.mc_mis(n, use_power_heuristic=False)
        results.append(("MIS (средняя плотность)", I_estimate, abs(I_estimate - I_true), delta_I))
        I_estimate = monte_carlo.mc_mis(n, use_power_heuristic=True)
        results.append(("MIS (средний квадрат)", I_estimate, abs(I_estimate - I_true), delta_I))
        I_estimate = monte_carlo.mc_russian_roulette(n, R=0.5)
        results.append(("Русская рулетка R=0.5", I_estimate, abs(I_estimate - I_true), delta_I))
        I_estimate = monte_carlo.mc_russian_roulette(n, R=0.75)
        results.append(("Русская рулетка R=0.75", I_estimate, abs(I_estimate - I_true), delta_I))
        I_estimate = monte_carlo.mc_russian_roulette(n, R=0.95)
        results.append(("Русская рулетка R=0.95", I_estimate, abs(I_estimate - I_true), delta_I))
        print_table(results, n)

if __name__ == "__main__":
    main()

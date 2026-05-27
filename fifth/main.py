import numpy as np
import trimesh

import math
import random
import time
import sys
import argparse
from datetime import datetime
from typing import Optional, Tuple

# ====================================================================
# Базовые математические структуры
# ====================================================================
class Vector3:
    """Трехмерный вектор с основными операциями."""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Vector3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> 'Vector3':
        return self * scalar

    def __truediv__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> 'Vector3':
        return Vector3(-self.x, -self.y, -self.z)

    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3') -> 'Vector3':
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    @property
    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> 'Vector3':
        norm = self.length
        if norm > 0:
            return self / norm
        return self

    def reflect(self, normal: 'Vector3') -> 'Vector3':
        return self - normal * (2 * self.dot(normal))

    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        if idx == 2:
            return self.z
        raise IndexError(f"Index {idx} out of range")

    @staticmethod
    def random_cosine_direction(normal: 'Vector3') -> 'Vector3':
        """Генерация случайного вектора по косинусному распределению."""
        r1 = random.random()
        r2 = random.random()
        phi = 2 * math.pi * r1
        sin_theta = math.sqrt(r2)
        cos_theta = math.sqrt(1 - r2)
        x = math.cos(phi) * sin_theta
        y = math.sin(phi) * sin_theta
        z = cos_theta
        if abs(normal.x) > abs(normal.y):
            tangent = Vector3(normal.z, 0, -normal.x).normalize()
        else:
            tangent = Vector3(0, -normal.z, normal.y).normalize()
        bitangent = normal.cross(tangent)
        return (tangent * x + bitangent * y + normal * z).normalize()

# ====================================================================
# Объекты свойств цвета и света
# ====================================================================
class Spectrum:
    """Спектральный цвет с RGB-компонентами."""
    def __init__(self, r: float = 0.0, g: float = 0.0, b: float = 0.0):
        self.r = float(r)
        self.g = float(g)
        self.b = float(b)

    def __repr__(self) -> str:
        return f"Spectrum({self.r:.4f}, {self.g:.4f}, {self.b:.4f})"

    def __add__(self, other: 'Spectrum') -> 'Spectrum':
        return Spectrum(self.r + other.r, self.g + other.g, self.b + other.b)

    def __sub__(self, other: 'Spectrum') -> 'Spectrum':
        return Spectrum(self.r - other.r, self.g - other.g, self.b - other.b)

    def __mul__(self, other) -> 'Spectrum':
        if isinstance(other, Spectrum):
            return Spectrum(self.r * other.r, self.g * other.g, self.b * other.b)
        return Spectrum(self.r * other, self.g * other, self.b * other)

    def __rmul__(self, scalar: float) -> 'Spectrum':
        return self * scalar

    def __truediv__(self, scalar: float) -> 'Spectrum':
        return Spectrum(self.r / scalar, self.g / scalar, self.b / scalar)

    @property
    def luminance(self) -> float:
        """Вычисление яркости цвета."""
        return 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b

    @staticmethod
    def black() -> 'Spectrum':
        """Черный цвет."""
        return Spectrum(0, 0, 0)

    @staticmethod
    def white() -> 'Spectrum':
        """Белый цвет."""
        return Spectrum(1, 1, 1)

class Ray:
    """Луч с начальной точкой и направлением."""
    def __init__(self, origin: Vector3, direction: Vector3):
        self.origin = origin
        self.direction = direction.normalize()

class Material:
    """Оптические свойства материала поверхности"""
    def __init__(
        self,
        diffuse: Spectrum,
        specular: Spectrum,
        emission: Spectrum = None,
        name: str = ""
    ):
        self.diffuse = diffuse
        self.specular = specular
        self.emission = emission if emission else Spectrum.black()
        self.name = name
        # проверка физической корректности материала и исправление если нужно
        self._validate_materials()

    def is_emissive(self) -> bool:
        """Проверка, является ли материал источником света"""
        return self.emission.r > 0 or self.emission.g > 0 or self.emission.b > 0

    def total_reflectance(self) -> Spectrum:
        """Суммарная отражательная способность"""
        return Spectrum(
            self.diffuse.r + self.specular.r,
            self.diffuse.g + self.specular.g,
            self.diffuse.b + self.specular.b
        )

    def is_physical(self) -> bool:
        """Проверка, что сумма коэффициентов отражения не превышает 1"""
        total = self.total_reflectance()
        return total.r <= 1.0 and total.g <= 1.0 and total.b <= 1.0

    def _validate_materials(self) -> None:
        """Проверка, что все материалы физически корректны"""
        if self.is_physical():
            print(f"Материал '{self.name}' физически корректен")
            return
        print(
            f"Материал '{self.name}' НЕ физически корректен! Сумма коэффициентов: {self.total_reflectance()}"
        )
        self._make_physical()
        print(f"Исправлено: новые коэффициенты diff={self.diffuse}, spec={self.specular}")

    def _make_physical(self, max_reflectance: float = 0.99) -> None:
        """Исправление на физически корректную версию материала"""
        total = self.total_reflectance()
        # Масштабируем каждый канал независимо
        scale_r = max_reflectance / total.r if total.r > 1.0 else 1.0
        scale_g = max_reflectance / total.g if total.g > 1.0 else 1.0
        scale_b = max_reflectance / total.b if total.b > 1.0 else 1.0
        # Применяем масштабирование
        new_diffuse = Spectrum(
            self.diffuse.r * scale_r,
            self.diffuse.g * scale_g,
            self.diffuse.b * scale_b
        )
        new_specular = Spectrum(
            self.specular.r * scale_r,
            self.specular.g * scale_g,
            self.specular.b * scale_b
        )
        self.diffuse = new_diffuse
        self.specular = new_specular

# ====================================================================
# Геометрические объекты
# ====================================================================
class IntersectionInfo:
    """Информация о пересечении луча с поверхностью."""
    def __init__(
        self,
        t: float,
        point: Vector3,
        normal: Vector3,
        material: Material,
        emission: Spectrum,
        uv: Tuple[float, float]
    ):
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material
        self.emission = emission
        self.uv = uv

    @property
    def is_light_source(self) -> bool:
        """Проверка, является ли поверхность источником света"""
        return self.emission.r > 0 or self.emission.g > 0 or self.emission.b > 0

class Triangle:
    """Треугольный примитив с интерполяцией нормалей"""
    def __init__(
        self,
        v0: Vector3,
        v1: Vector3,
        v2: Vector3,
        n0: Vector3,
        n1: Vector3,
        n2: Vector3,
        material: Material,
        emission: Spectrum = None
    ):
        self.vertices = (v0, v1, v2)
        self.normals = (n0, n1, n2)
        self.material = material
        self.emission = emission if emission else Spectrum.black()
        # Площадь треугольника (для выборки источников света)
        edge1 = v1 - v0
        edge2 = v2 - v0
        self.area = edge1.cross(edge2).length / 2.0

# ====================================================================
# Камера и проекция
# ====================================================================
class Camera:
    """Точечная камера с перспективной проекцией"""
    def __init__(
        self,
        position: Vector3,
        target: Vector3,
        up: Vector3,
        fov_degrees: float,
        width: int,
        height: int
    ):
        self.position = position
        self.forward = (target - position).normalize()
        self.right = self.forward.cross(up).normalize()
        self.up = self.right.cross(self.forward).normalize()
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        # Вычисление размеров пикселя
        fov_rad = math.radians(fov_degrees)
        half_height = math.tan(fov_rad / 2.0)
        half_width = self.aspect_ratio * half_height
        self.pixel_width = half_width * 2 / width
        self.pixel_height = half_height * 2 / height
        # Векторы для плоскости изображения
        self.screen_center = self.position + self.forward
        self.screen_u = self.right * half_width
        self.screen_v = self.up * half_height

    def generate_ray(
        self,
        x: int,
        y: int,
        offset_x: float = 0.0,
        offset_y: float = 0.0
    ) -> Ray:
        """Генерация луча через заданный пиксель"""
        # Координаты с учетом антиалиасинга
        pixel_x = x + offset_x
        pixel_y = y + offset_y
        # Вычисление точки на плоскости изображения
        u = (pixel_x / self.width - 0.5) * 2.0
        v = (0.5 - pixel_y / self.height) * 2.0
        point_on_screen = (
            self.screen_center +
            self.screen_u * u +
            self.screen_v * v
        )
        direction = (point_on_screen - self.position).normalize()
        return Ray(self.position, direction)

# ====================================================================
# Сцена и источники света
# ====================================================================
class Scene:
    """Трехмерная сцена с использованием trimesh для ускорения"""
    def __init__(self):
        self.triangles = []
        self.lights = []
        self.trimesh_cache = None  # Кэшированная сцена для ускорения

    def add_triangle(self, triangle: Triangle):
        """Добавление треугольника в сцену"""
        self.triangles.append(triangle)
        if triangle.emission.r > 0 or triangle.emission.g > 0 or triangle.emission.b > 0:
            self.lights.append(triangle)
        # Сбрасываем кэш при добавлении нового треугольника
        self.trimesh_cache = None

    def add_sphere(
        self,
        center: Vector3,
        radius: float,
        material: Material,
        subdivisions=1
    ):
        """Добавление сфер (аппроксимированных треугольниками)"""
        # Вершины икосаэдра
        t = (1.0 + math.sqrt(5.0)) / 2.0
        base_vertices = [
            Vector3(-1, t, 0), Vector3(1, t, 0), Vector3(-1, -t, 0), Vector3(1, -t, 0),
            Vector3(0, -1, t), Vector3(0, 1, t), Vector3(0, -1, -t), Vector3(0, 1, -t),
            Vector3(t, 0, -1), Vector3(t, 0, 1), Vector3(-t, 0, -1), Vector3(-t, 0, 1)
        ]
        base_vertices = [v.normalize() for v in base_vertices]
        faces = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
        ]
        for face in faces:
            self._subdivide(
                base_vertices[face[0]],
                base_vertices[face[1]],
                base_vertices[face[2]],
                center,
                radius,
                material,
                subdivisions
            )

    def _subdivide(
        self,
        v1: Vector3,
        v2: Vector3,
        v3: Vector3,
        center: Vector3,
        radius: float,
        material: Material,
        level: int
    ) -> None:
        """Рекурсивное подразделение треугольников для аппроксимации сферы"""
        if level == 0:
            p0 = center + v1 * radius
            p1 = center + v2 * radius
            p2 = center + v3 * radius
            n0 = v1.normalize()
            n1 = v2.normalize()
            n2 = v3.normalize()
            self.add_triangle(Triangle(p0, p1, p2, n0, n1, n2, material))
        else:
            v12 = (v1 + v2).normalize()
            v23 = (v2 + v3).normalize()
            v31 = (v3 + v1).normalize()
            self._subdivide(v1, v12, v31, center, radius, material, level - 1)
            self._subdivide(v2, v23, v12, center, radius, material, level - 1)
            self._subdivide(v3, v31, v23, center, radius, material, level - 1)
            self._subdivide(v12, v23, v31, center, radius, material, level - 1)

    def _build_trimesh_scene(self):
        """Построение сцены trimesh (один раз при первом вызове)"""
        if self.trimesh_cache is not None:
            return self.trimesh_cache
        vertices = []
        faces = []
        face_to_triangle = []
        for i, tri in enumerate(self.triangles):
            v0, v1, v2 = tri.vertices
            start_idx = len(vertices)
            vertices.append([v0.x, v0.y, v0.z])
            vertices.append([v1.x, v1.y, v1.z])
            vertices.append([v2.x, v2.y, v2.z])
            faces.append([start_idx, start_idx + 1, start_idx + 2])
            face_to_triangle.append(i)
        # Создаем меш trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # Сохраняем структуру с сопоставлениями
        self.trimesh_cache = {
            'mesh': mesh,
            'face_to_triangle': face_to_triangle,
            'vertices': vertices,
            'faces': faces
        }
        return self.trimesh_cache

    def intersect(self, ray: Ray) -> Optional[IntersectionInfo]:
        """Оптимизированный поиск пересечений с использованием trimesh"""
        # Используем trimesh для быстрого поиска пересечений
        scene_data = self._build_trimesh_scene()
        mesh = scene_data['mesh']
        face_to_triangle = scene_data['face_to_triangle']
        # Преобразуем луч для trimesh
        ray_origin = np.array([[ray.origin.x, ray.origin.y, ray.origin.z]], dtype=np.float64)
        ray_direction = np.array([[ray.direction.x, ray.direction.y, ray.direction.z]], dtype=np.float64)
        # Используем оптимизированный поиск trimesh
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origin,
            ray_directions=ray_direction,
            multiple_hits=False
        )
        if len(locations) == 0:
            return None
        # Нашли пересечение - берем первый (ближайший)
        triangle_idx = face_to_triangle[index_tri[0]]
        triangle = self.triangles[triangle_idx]
        point = Vector3(locations[0][0], locations[0][1], locations[0][2])
        # Вычисляем барицентрические координаты
        A, B, C = triangle.vertices
        u, v, w = self.compute_barycentric_coordinates(point, A, B, C)
        # Интерполируем нормаль
        normal = (
            triangle.normals[0] * u +
            triangle.normals[1] * v +
            triangle.normals[2] * w
        )
        norm_length = normal.length
        if norm_length < 1e-8:
            # Вырожденная нормаль, используем нормаль из треугольника
            normal = triangle.normals[0]
        else:
            normal = normal / norm_length
        # Проверяем, что луч приходит с лицевой стороны треугольника
        if ray.direction.dot(normal) >= 0:
            # Луч приходит с тыльной стороны, игнорируем пересечение
            return None
        # Вычисляем расстояние t
        t = (point - ray.origin).length
        # Возвращаем информацию о пересечении
        return IntersectionInfo(
            t=t,
            point=point,
            normal=normal,
            material=triangle.material,
            emission=triangle.emission,
            uv=(v, w)  # координаты для вершин B и C
        )

    @staticmethod
    def compute_barycentric_coordinates(P: Vector3, A: Vector3, B: Vector3, C: Vector3) -> Tuple[float, float, float]:
        """Вычисление барицентрических координат точки P относительно треугольника ABC"""
        # Решение системы уравнений через проекции
        v0 = B - A
        v1 = C - A
        v2 = P - A
        # Вычисление через скалярные произведения
        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d11 = v1.dot(v1)
        d20 = v2.dot(v0)
        d21 = v2.dot(v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-8:
            return 1.0, 0.0, 0.0
        # Вычисление координат
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        # Обрезаем до [0,1] для устойчивости
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        w = max(0.0, min(1.0, w))
        # Нормализуем, чтобы сумма была равна 1
        total = u + v + w
        if total > 0:
            u /= total
            v /= total
            w /= total
        return u, v, w

# ====================================================================
# Трассировщик путей
# ====================================================================
class PathTracer:
    """Трассировщик путей"""
    def __init__(
        self,
        scene: Scene,
        camera: Camera,
        width: int,
        height: int,
        samples_per_pixel: int = 16,
        max_depth: int = 5
    ):
        self.scene = scene
        self.camera = camera
        self.width = width
        self.height = height
        self.samples = samples_per_pixel
        self.max_depth = max_depth
        # Инициализация изображения
        self.image = np.zeros((height, width, 3), dtype=np.float32)

    def trace_path(self, ray: Ray, depth: int = 0) -> Spectrum:
        """Основной алгоритм трассировки пути"""
        if depth >= self.max_depth:
            return Spectrum.black()
        hit = self.scene.intersect(ray)
        if not hit:
            return Spectrum.black()
        # Если попали на источник света - возвращаем его излучение
        if hit.is_light_source:
            return hit.emission
        material = hit.material
        point = hit.point
        normal = hit.normal
        # Русская рулетка
        if depth > 3:  # Начинаем применять после 3-го отскока
            survival_prob = 0.8
            if random.random() > survival_prob:
                return Spectrum.black()
        else:
            survival_prob = 1.0
        # Выбор типа отражения (диффузное или зеркальное)
        diffuse_weight = material.diffuse.luminance
        specular_weight = material.specular.luminance
        total_weight = diffuse_weight + specular_weight
        if total_weight < 1e-6:
            return Spectrum.black()
        # Нормализуем вероятности
        diffuse_prob = diffuse_weight / total_weight
        specular_prob = specular_weight / total_weight
        # Выбор случайного направления
        if random.random() < diffuse_prob:
            # Диффузное отражение (Lambert) - выбрано с вероятностью diffuse_prob
            new_direction = Vector3.random_cosine_direction(normal)
            pdf = max(1e-8, new_direction.dot(normal) / math.pi)
            if pdf < 1e-12:
                return Spectrum.black()
            new_ray = Ray(point + normal * 1e-4, new_direction)
            cosine = abs(new_direction.dot(normal))
            incoming = self.trace_path(new_ray, depth + 1)
            brdf = material.diffuse / math.pi
            # Косвенное освещение (importance sampling по BRDF)
            # Делим на вероятность выбора диффузного отражения
            indirect_light = incoming * brdf * cosine / (pdf * diffuse_prob)
            # Прямое освещение (next event estimation)
            # Также делим на вероятность выбора диффузного отражения
            direct_light = self.sample_direct_lighting(point, normal) * brdf / diffuse_prob
            result = indirect_light + direct_light
        else:
            # Зеркальное отражение - выбрано с вероятностью specular_prob
            reflected = ray.direction.reflect(normal).normalize()
            new_ray = Ray(point + normal * 1e-4, reflected)
            incoming = self.trace_path(new_ray, depth + 1)
            # Для идеального зеркала BRDF = specular * delta-функция
            # PDF для зеркального отражения = 1 (детерминированное направление)
            # Делим на вероятность выбора зеркального отражения
            result = incoming * material.specular / specular_prob
        # Масштабируем по вероятности выживания (русская рулетка)
        if depth > 3:
            result = result / survival_prob
        return result

    def sample_direct_lighting(self, point: Vector3, normal: Vector3) -> Spectrum:
        """Выборка прямого освещения (один сэмпл)"""
        if not self.scene.lights:
            return Spectrum.black()
        # Выбираем случайный источник света
        light = random.choice(self.scene.lights)
        # Выбор случайной точки на источнике света
        u = random.random()
        v = random.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        w = 1 - u - v
        # Вычисляем точку на источнике света
        light_point = (
            light.vertices[0] * w +
            light.vertices[1] * u +
            light.vertices[2] * v
        )
        to_light = light_point - point
        distance = to_light.length
        if distance <= 1e-8:
            return Spectrum.black()
        light_dir = to_light / distance
        cos_surface = abs(normal.dot(light_dir))
        if cos_surface <= 0:
            return Spectrum.black()
        # Проверка видимости
        eps = 1e-4
        origin = point + normal * eps
        shadow_ray = Ray(origin, light_dir)
        shadow_hit = self.scene.intersect(shadow_ray)
        if shadow_hit and shadow_hit.t < distance - 1e-4:
            return Spectrum.black()
        # Вычисление геометрического фактора
        light_normal = (
            light.normals[0] * w +
            light.normals[1] * u +
            light.normals[2] * v
        ).normalize()
        cos_light = abs(light_normal.dot(-light_dir))
        if cos_light <= 0:
            return Spectrum.black()
        edge1 = light.vertices[1] - light.vertices[0]
        edge2 = light.vertices[2] - light.vertices[0]
        light_area = edge1.cross(edge2).length / 2.0
        if light_area <= 0:
            return Spectrum.black()
        # PDF для выборки света
        light_pdf = 1.0 / (len(self.scene.lights) * light_area)
        # Формула освещения
        geometry_factor = cos_surface * cos_light / (distance * distance)
        if geometry_factor <= 0:
            return Spectrum.black()
        # Возвращаем вклад сэмпла
        return light.emission * geometry_factor / light_pdf

    def render(self) -> np.ndarray:
        """Основной метод рендеринга"""
        print(f"Начало рендеринга: {self.width}x{self.height}")
        print(f"Сэмплов на пиксель: {self.samples}")
        print(f"Глубина трассировки: {self.max_depth}")
        print(f"Треугольников в сцене: {len(self.scene.triangles)}")
        start_time = time.time()
        total_pixels = self.width * self.height
        pixel_counter = 0
        for y in range(self.height):
            for x in range(self.width):
                color = Spectrum.black()
                # Мультисемплирование на уровне пикселя (антиалиасинг)
                for _ in range(self.samples):
                    # Антиалиасинг
                    offset_x = random.random() - 0.5
                    offset_y = random.random() - 0.5
                    ray = self.camera.generate_ray(x, y, offset_x, offset_y)
                    path_color = self.trace_path(ray)
                    color += path_color
                # Усреднение по количеству сэмплов
                color = color / self.samples
                self.image[y, x] = [color.r, color.g, color.b]
                pixel_counter += 1
                if pixel_counter % 400 == 0:
                    progress = pixel_counter / total_pixels * 100
                    elapsed = time.time() - start_time
                    print(
                        f"\rПрогресс: {progress:.1f}% ({pixel_counter}/{total_pixels}), время: {elapsed:.1f}с",
                        end="", flush=True
                    )
        elapsed = time.time() - start_time
        print(f"\nРендеринг завершен за {elapsed:.2f} секунд")
        return self.image

# ====================================================================
# Утилиты обработки изображений
# ====================================================================
def tonemap(
    image_data: np.ndarray,
    exposure: float = 1.0,
    gamma: float = 2.2
) -> np.ndarray:
    """ACES тонмаппинг"""
    image_data = image_data * exposure
    # ACES approximation
    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14
    image_data = (image_data * (A * image_data + B)) / (image_data * (C * image_data + D) + E)
    # Отсечение и гамма
    image_data = np.clip(image_data, 0.0, 1.0)
    image_data = np.power(image_data, 1.0 / gamma)
    return image_data

def save_ppm(filename: str, image_data: np.ndarray):
    """Сохранение изображения в формате PPM"""
    height, width, _ = image_data.shape
    # Конвертация в 8-битный формат
    image_8bit = (np.clip(image_data, 0, 1) * 255).astype(np.uint8)
    with open(filename, 'wb') as f:
        # Заголовок PPM
        header = f"P6\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        # Данные пикселей
        for y in range(height):
            for x in range(width):
                r, g, b = image_8bit[y, x]
                f.write(bytes([r, g, b]))
    print(f"Изображение сохранено: {filename}")

# ====================================================================
# Построение конкретной сцены
# ====================================================================
class CornellBoxScene:
    def __init__(self):
        self.red_material = Material(
            diffuse=Spectrum(0.65, 0.05, 0.05),
            specular=Spectrum(0.0, 0.0, 0.0),
            name="Red Material"
        )
        self.green_material = Material(
            diffuse=Spectrum(0.12, 0.45, 0.15),
            specular=Spectrum(0.0, 0.0, 0.0),
            name="Green Material"
        )
        self.white_material = Material(
            diffuse=Spectrum(0.9, 0.9, 0.9),
            specular=Spectrum(0.0, 0.0, 0.0),
            name="White Material"
        )
        self.mirror_white_material = Material(
            diffuse=Spectrum(0.9, 0.9, 0.9),
            specular=Spectrum(0.1, 0.1, 0.1),
            name="Mirror White Material"
        )
        self.mirror_material = Material(
            diffuse=Spectrum(0.01, 0.01, 0.01),
            specular=Spectrum(0.99, 0.99, 0.99),
            name="Mirror Material"
        )
        self.light = Material(
            diffuse=Spectrum(0.0, 0.0, 0.0),
            specular=Spectrum(0.0, 0.0, 0.0),
            emission=Spectrum(6.0,  6.0, 6.0),
            name="Light"
        )
        self.scene = self._create_cornell_box()

    def _create_cornell_box(self) -> Scene:
        """Создание сцены Корнелл бокс"""
        scene = Scene()
        # Векторы нормалей
        up = Vector3(0, 1, 0)
        down = Vector3(0, -1, 0)
        left = Vector3(1, 0, 0)
        right = Vector3(-1, 0, 0)
        back = Vector3(0, 0, -1)
        # Пол (белый) - нормали вверх
        scene.add_triangle(Triangle(
            Vector3(-1, -1, -1), Vector3(1, -1, -1), Vector3(-1, -1, 1),
            up, up, up, self.white_material
        ))
        scene.add_triangle(Triangle(
            Vector3(1, -1, -1), Vector3(1, -1, 1), Vector3(-1, -1, 1),
            up, up, up, self.white_material
        ))
        # Потолок (белый) - нормали вниз
        scene.add_triangle(Triangle(
            Vector3(-1, 1, -1), Vector3(-1, 1, 1), Vector3(1, 1, -1),
            down, down, down, self.white_material
        ))
        scene.add_triangle(Triangle(
            Vector3(1, 1, -1), Vector3(-1, 1, 1), Vector3(1, 1, 1),
            down, down, down, self.white_material
        ))
        # Левая стена (красная) - нормаль вправо
        scene.add_triangle(Triangle(
            Vector3(-1, -1, -1), Vector3(-1, -1, 1), Vector3(-1, 1, -1),
            left, left, left, self.red_material
        ))
        scene.add_triangle(Triangle(
            Vector3(-1, 1, -1), Vector3(-1, -1, 1), Vector3(-1, 1, 1),
            left, left, left, self.red_material
        ))
        # Правая стена (зеленая) - нормаль влево
        scene.add_triangle(Triangle(
            Vector3(1, -1, -1), Vector3(1, 1, -1), Vector3(1, -1, 1),
            right, right, right, self.green_material
        ))
        scene.add_triangle(Triangle(
            Vector3(1, 1, -1), Vector3(1, 1, 1), Vector3(1, -1, 1),
            right, right, right, self.green_material
        ))
        # Задняя стена (белая-слегка зеркальная) - нормаль вперед
        scene.add_triangle(Triangle(
            Vector3(-1, -1, 1), Vector3(1, -1, 1), Vector3(-1, 1, 1),
            back, back, back, self.mirror_white_material
        ))
        scene.add_triangle(Triangle(
            Vector3(1, -1, 1), Vector3(1, 1, 1), Vector3(-1, 1, 1),
            back, back, back, self.mirror_white_material
        ))
        # Источник света (на потолке)
        light_size = 0.5
        scene.add_triangle(Triangle(
            Vector3(-light_size, 0.999, -light_size),
            Vector3(-light_size, 0.999, light_size),
            Vector3(light_size, 0.999, -light_size),
            down, down, down, self.light, self.light.emission
        ))
        scene.add_triangle(Triangle(
            Vector3(light_size, 0.999, -light_size),
            Vector3(-light_size, 0.999, light_size),
            Vector3(light_size, 0.999, light_size),
            down, down, down, self.light, self.light.emission
        ))
        # добавление объектов на сцену
        scene.add_sphere(
            Vector3(0.4, -0.7, -0.2),
            0.3,
            self.white_material,
            1
        )
        scene.add_sphere(
            Vector3(-0.4, 0.0, 0.2),
            0.2,
            self.mirror_material,
            1
        )
        return scene

# ====================================================================
# Интерфейс командной строки
# ====================================================================
def main():
    """Основная функция программы."""
    parser = argparse.ArgumentParser(
        description="Path Tracer with Global Illumination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Быстрый тест
  python path_tracer.py --width 200 --height 200 --samples 4 --max-depth 3

  # Среднее качество
  python path_tracer.py --width 500 --height 500 --samples 16 --max-depth 5

  # Высокое качество
  python path_tracer.py --width 600 --height 600 --samples 24 --max-depth 8
        """
    )
    parser.add_argument(
        "--width", type=int, default=500, help="Ширина изображения"
    )
    parser.add_argument(
        "--height", type=int, default=500, help="Высота изображения"
    )
    parser.add_argument(
        "--samples", type=int, default=16, help="Количество сэмплов на пиксель"
    )
    parser.add_argument(
        "--max-depth", type=int, default=8, help="Максимальная глубина трассировки"
    )
    parser.add_argument(
        "--fov", type=float, default=45.0, help="Поле зрения камеры"
    )
    parser.add_argument(
        "--exposure", type=float, default=1.0, help="Экспозиция для тонмэппинга"
    )
    parser.add_argument(
        "--gamma", type=float, default=2.2, help="Гамма-коррекция"
    )
    parser.add_argument(
        "--output", type=str,
        default=f"render-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.ppm",
        help="Имя выходного файла"
    )
    args = parser.parse_args()
    print(f"Разрешение:          {args.width}x{args.height}")
    print(f"Сэмплов на пиксель:  {args.samples}")
    print(f"Глубина трассировки: {args.max_depth}")
    print(f"Поле зрения камеры:  {args.fov}")
    print(f"Экспозиция:          {args.exposure}")
    print(f"Гамма-коррекция:     {args.gamma}")
    # Создание сцены
    print("\nСоздание сцены...")
    cornell_box_scene = CornellBoxScene()
    scene = cornell_box_scene.scene
    print(f"Треугольников:    {len(scene.triangles)}")
    print(f"Источников света: {len(scene.lights)}")
    # Создание камеры и трассировщика
    camera_pos = Vector3(0, 0, -3)
    camera_target = Vector3(0, 0, 0)
    camera_up = Vector3(0, 1, 0)
    camera = Camera(
        camera_pos,
        camera_target,
        camera_up,
        args.fov,
        args.width,
        args.height
    )
    tracer = PathTracer(
        scene,
        camera,
        args.width,
        args.height,
        args.samples,
        args.max_depth
    )
    # Рендеринг
    print("\nНачало рендеринга...")
    image_data = tracer.render()
    # Тонмэппинг
    print("\nПрименение тонмэппинга...")
    tone_mapped = tonemap(image_data, args.exposure, args.gamma)
    # Сохранение результата
    save_ppm(args.output, tone_mapped)
    print(f"\nРендеринг завершен!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nРендеринг прерван пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

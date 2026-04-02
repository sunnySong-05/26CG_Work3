import taichi as ti
import numpy as np

# 初始化与显存预分配
ti.init(arch=ti.gpu)

RES = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

# 分配GPU缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_SEGMENTS + 1))
gui_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS))

# --- 算法部分 ---

# Bezier: De Casteljau 递归算法
def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    new_points = []
    for i in range(len(points) - 1):
        x = (1 - t) * points[i][0] + t * points[i + 1][0]
        y = (1 - t) * points[i][1] + t * points[i + 1][1]
        new_points.append([x, y])
    return de_casteljau(new_points, t)

# B-Spline: 均匀三次B样条矩阵实现
def b_spline_point(p0, p1, p2, p3, t):
    # 均匀三次B样条基矩阵
    M = np.array([
        [-1,  3, -3, 1],
        [ 3, -6,  3, 0],
        [-3,  0,  3, 0],
        [ 1,  4,  1, 0]
    ]) / 6.0
    T = np.array([t**3, t**2, t, 1])
    P = np.array([p0, p1, p2, p3])
    # 计算 P(t) = T * M * P
    return T @ M @ P

# --- GPU 渲染内核 ---

@ti.kernel
def draw_curve_kernel(n: ti.i32, color_r: ti.f32, color_g: ti.f32, color_b: ti.f32):
    for i in range(n):
        pos = curve_points_field[i]
        
        # 归一化坐标转像素坐标 (浮点数，保留精度)
        f_px = pos[0] * float(RES)
        f_py = pos[1] * float(RES)
        
        # 核心：反走样渲染 (AA)
        # 遍历该点周围的 3x3 像素领域
        base_x = int(f_px)
        base_y = int(f_py)
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px, py = base_x + dx, base_y + dy
                
                if 0 <= px < RES and 0 <= py < RES:
                    # 计算像素中心点到精确曲线点的距离
                    dist = ti.sqrt((float(px) + 0.5 - f_px)**2 + (float(py) + 0.5 - f_py)**2)
                    
                    # 距离衰减模型：采用简单的线性径向衰减或高斯模拟
                    # 这里使用 1.5 像素半径内的线性衰减，增强平滑感
                    intensity = ti.max(0.0, 1.0 - dist / 1.5)
                    
                    # 颜色累加（使用原子加法以处理多点重叠）
                    ti.atomic_add(pixels[px, py], ti.Vector([color_r, color_g, color_b]) * intensity)

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

# --- 主逻辑 ---

def main():
    window = ti.ui.Window("Bézier vs B-Spline (Press Z/B to switch)", (RES, RES))
    canvas = window.get_canvas()
    control_points = []
    mode = 'bezier' # 初始模式

    while window.running:
        # 事件处理
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    mouse = window.get_cursor_pos()
                    control_points.append([mouse[0], mouse[1]])
            elif window.event.key == 'c':
                control_points = []
            elif window.event.key == 'z': # 切换到 Bezier
                mode = 'bezier'
            elif window.event.key == 'b': # 切换到 B-Spline
                mode = 'b-spline'

        clear_pixels()

        # 计算曲线数据
        if len(control_points) >= 2:
            curve_data = []
            
            if mode == 'bezier':
                # Bezier 模式：全局计算
                for i in range(NUM_SEGMENTS + 1):
                    t = i / NUM_SEGMENTS
                    p = de_casteljau(control_points, t)
                    curve_data.append(p)
                color = (0.0, 1.0, 0.0) # 绿色
                
            elif mode == 'b-spline':
                # B-Spline 模式：分段计算
                n = len(control_points)
                if n >= 4:
                    # n个点有 n-3 个线段
                    segments = n - 3
                    pts_per_seg = NUM_SEGMENTS // segments
                    for s in range(segments):
                        p0, p1, p2, p3 = [control_points[s+i] for i in range(4)]
                        for i in range(pts_per_seg + 1):
                            t = i / pts_per_seg
                            p = b_spline_point(p0, p1, p2, p3, t)
                            curve_data.append(p)
                else:
                    # 点不够时，暂时退化为折线或空
                    curve_data = control_points
                color = (0.3, 0.6, 1.0) # 蓝色
            
            # 渲染曲线
            if len(curve_data) > 0:
                # 填充 field (动态截断或补足)
                np_curve = np.array(curve_data, dtype=np.float32)
                if len(np_curve) > NUM_SEGMENTS + 1:
                    np_curve = np_curve[:NUM_SEGMENTS + 1]
                
                curve_points_field.from_numpy(np.zeros((NUM_SEGMENTS+1, 2), dtype=np.float32)) # 先清空
                curve_points_field.from_numpy(np_curve)
                draw_curve_kernel(len(np_curve), color[0], color[1], color[2])

        # GUI 绘制
        gui_pts_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
        if len(control_points) > 0:
            gui_pts_np[:len(control_points)] = np.array(control_points)
        gui_points_field.from_numpy(gui_pts_np)

        canvas.set_image(pixels)
        if len(control_points) >= 2:
            canvas.lines(gui_points_field, width=0.001, color=(0.3, 0.3, 0.3))
        canvas.circles(gui_points_field, radius=0.008, color=(1.0, 0.2, 0.2))

        window.show()

if __name__ == "__main__":
    main()
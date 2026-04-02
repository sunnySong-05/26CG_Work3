import taichi as ti
import numpy as np

#初始化与显存预分配
ti.init(arch=ti.gpu)

RES = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

# 分配GPU缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_SEGMENTS + 1))
gui_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS))

#实现De Casteljau算法
#递归计算贝塞尔曲线在参数t处的坐标
#points:当前层的控制点列表
def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    
    new_points = []
    for i in range(len(points) - 1):
        # 线性插值公式:P=(1-t)P_i+t*P_{i+1}
        x = (1 - t) * points[i][0] + t * points[i + 1][0]
        y = (1 - t) * points[i][1] + t * points[i + 1][1]
        new_points.append([x, y])
    
    return de_casteljau(new_points, t)

#编写GPU绘制内核
@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        pos = curve_points_field[i]
        
        #坐标映射：归一化浮点数到物理像素索引
        px = int(pos[0] * RES)
        py = int(pos[1] * RES)
        
        #越界检查与点亮像素
        if 0 <= px < RES and 0 <= py < RES:
            pixels[px, py] = ti.Vector([0.0, 1.0, 0.0])

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

#主逻辑与交互
def main():
    window = ti.ui.Window("Bézier Curve Rasterization", (RES, RES))
    canvas = window.get_canvas()
    control_points = []

    while window.running:
        mouse = window.get_cursor_pos()
        
        #监听鼠标左键点击
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    control_points.append([mouse[0], mouse[1]])
            
            #监听键盘 C 键清空
            elif window.event.key == 'c':
                control_points = []

        #每帧清空像素缓冲区
        clear_pixels()

        #计算与绘制
        if len(control_points) >= 2:
            #在 CPU 端计算曲线点 (Batching 准备)
            curve_data = []
            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                p = de_casteljau(control_points, t)
                curve_data.append(p)
            
            #批量拷贝到 GPU 并调用 Kernel 绘制
            curve_points_field.from_numpy(np.array(curve_data, dtype=np.float32))
            draw_curve_kernel(NUM_SEGMENTS + 1)

        #绘制交互控制点（对象池）
        #创建一个隐藏在屏幕外的初始数组
        gui_pts_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
        if len(control_points) > 0:
            gui_pts_np[:len(control_points)] = np.array(control_points)
        
        gui_points_field.from_numpy(gui_pts_np)

        canvas.set_image(pixels) #显示光栅化后的像素
        
        if len(control_points) >= 2:
            #绘制控制多边形（灰色折线）
            canvas.lines(gui_points_field, width=0.002, color=(0.5, 0.5, 0.5))
        
        #绘制控制点（红色圆点）
        canvas.circles(gui_points_field, radius=0.01, color=(1.0, 0.0, 0.0))

        window.show()

if __name__ == "__main__":
    main()
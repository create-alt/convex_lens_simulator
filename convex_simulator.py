import numpy as np
import matplotlib.pyplot as plt

def simulate_lens(r=100, n=1.5, d=80, num_rays=7):
    """
    レポートの定式化に基づく凸レンズシミュレータ
    r: 円の半径
    n: ガラスの相対屈折率
    d: 円の中心の原点からの距離 (レンズの厚みを決定)
    """
    # 描画設定
    plt.figure(figsize=(10, 6))
    
    # 1. レンズ形状の描画 (Vesica Piscis)
    # 左側の面（右側の円の一部）と右側の面（左側の円の一部）
    y_range = np.sqrt(r**2 - d**2)
    y_vals = np.linspace(-y_range, y_range, 400)
    x_left = d - np.sqrt(r**2 - y_vals**2)  # 左表面
    x_right = -d + np.sqrt(r**2 - y_vals**2) # 右表面
    
    plt.plot(x_left, y_vals, 'b-', label='Lens Surface')
    plt.plot(x_right, y_vals, 'b-')
    plt.fill_betweenx(y_vals, x_left, x_right, color='lightblue', alpha=0.3)

    # 2. 光線のシミュレーション
    ray_y_starts = np.linspace(-y_range*0.5, y_range*0.5, num_rays)
    
    for y1 in ray_y_starts:
        if np.isclose(y1, 0):
            plt.plot([-150, 150], [0, 0], 'r-', alpha=0.6)
            continue
            
        # --- 第1面での屈折 ---
        # 入射点 P1 (座標系は右側の円の中心を原点とする)
        x1_L = -np.sqrt(r**2 - y1**2)
        x1_g = x1_L + d # グローバル座標
        
        # 入射光（平行光）
        plt.plot([-150, x1_g], [y1, y1], 'r-', alpha=0.6)
        
        # 式(1): 入射角 theta
        theta = np.pi- np.arccos(x1_L / r)
        # print(np.degrees(theta))
        
        # 式(2): 方向ベクトル v'
        # 回転行列による計算
        cos_tn = np.cos(np.arcsin(np.sin(theta)/n))
        # print(np.degrees(np.arcsin(np.sin(theta)/n)))
        sin_tn = np.sin(np.arcsin(np.sin(theta)/n))
        v_px = x1_L * cos_tn - y1/abs(y1) * y1 * sin_tn
        v_py = y1/abs(y1) * x1_L * sin_tn + y1 * cos_tn
        
        # 式(3): レンズ内の直線の傾き
        slope_in = v_py / v_px
        
        # --- 第2面での屈折 ---
        # 第2面（左側の円）との交点 P2 を計算
        # (x+d)^2 + (m(x-x1)+y1)^2 = r^2 を解く
        m = slope_in
        # print(m)

        b = y1 - m*(d+r-abs(x1_L))
        # print(b)

        A = 1 + m**2
        B = 2 * b * m 
        C = b**2 - r**2
        
        discriminant = B**2 - 4*A*C
        # print(discriminant)
        if discriminant < 0: continue
        
        x2_g = (-B + np.sqrt(discriminant)) / (2*A) - d
        y2 = m * (x2_g - x1_g) + y1
        
        # レンズ内部の光線描画
        plt.plot([x1_g, x2_g], [y1, y2], 'r-', alpha=0.6)
        
        # 式(4): 出射時の入射角 phi
        # 左側の円の中心を原点とした座標 x'2
        x2_L = x2_g + d
        # ベクトル v' の大きさ（式中では r とされている）
        mag_v = np.sqrt(v_px**2 + v_py**2)
        phi = np.pi- np.arccos((x2_L * v_px + y2 * v_py) / r**2)
        # print(np.degrees(phi))
        
        # 式(5): 出射後の直線の傾き
        num_out = x2_L * np.sin(np.arcsin(np.sin(phi)*n)) + y1/abs(y1)*y2 * np.cos(np.arcsin(np.sin(phi)*n))
        den_out = -y1/abs(y1)*x2_L * np.cos(np.arcsin(np.sin(phi)*n)) + y2 * np.sin(np.arcsin(np.sin(phi)*n))
        slope_out = num_out / den_out
        
        # 出射光の描画
        x_end = 250
        y_end = slope_out * (x_end - x2_g) + y2
        plt.plot([x2_g, x_end], [y2, y_end], 'r-', alpha=0.6)

    # グラフの装飾
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.xlim(-150, 250)
    plt.ylim(-100, 100)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('')
    plt.grid(True)
    plt.show()

# 実行
simulate_lens()
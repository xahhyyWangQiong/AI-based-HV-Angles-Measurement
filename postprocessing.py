import cv2
import numpy as np

def pca_regression(image_ori, bbox, heatmap, type=cv2.DIST_WELSCH, constant=0):
    import cv2
    import numpy as np
    
    x1, y1, x2, y2 = bbox
    image = image_ori[y1:y2, x1:x2].copy()
    
    # 将 heatmap 转为二值化形式（参考函数中阈值设为0.5）
    heatmap = np.where(heatmap > 0.5, 1, 0).astype(np.uint8)  # heatmap shape: (channels, height, width)
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # 蓝、绿、红
    h, w, _ = image.shape
    n_heatmap = np.zeros([heatmap.shape[0], h, w], dtype=np.uint8)
    
    # 调整每个通道的 heatmap 尺寸
    for c in range(heatmap.shape[0]):
        n_heatmap[c] = cv2.resize(heatmap[c], (w, h))
    heatmap = n_heatmap

    directions = []  # 存储各通道的单位方向向量
    overlay = image.copy()
    
    # 遍历每个通道，利用 compute_pca_angle 计算角度，并基于角度绘制直线
    for c in range(heatmap.shape[0]):
        prob_map = heatmap[c].astype(np.float32)  # 保证计算时为浮点型
        # 若当前通道中前景像素数为0，则跳过
        if np.sum(prob_map) == 0:
            directions.append(None)
            continue

        # 计算主方向（角度，单位：弧度），此函数内部利用前景概率计算加权协方差矩阵
        angle, cov, eigvals, eigvecs = compute_pca_angle(prob_map)
        # 构造单位方向向量
        dir_vector = np.array([np.cos(angle), np.sin(angle)])
        directions.append(dir_vector)
        
        # 为绘制直线，需要计算当前通道前景概率的加权质心
        cx, cy, _, _ = compute_soft_coordinates(prob_map)
        
        # 根据方向判断直线绘制方式：若 sin(angle) 绝对值较大，则从图像顶部到底部计算交点，否则从左侧到右侧计算交点
        if abs(np.sin(angle)) > 0.01:
            # 求 y=0 和 y=h 时的 t 参数
            t_top = -cy / np.sin(angle)
            t_bottom = (h - cy) / np.sin(angle)
            x_top = cx + t_top * np.cos(angle)
            x_bottom = cx + t_bottom * np.cos(angle)
            pt1 = (int(round(x_top)), 0)
            pt2 = (int(round(x_bottom)), h)
        else:
            # 当直线接近水平时，以左右边界做交点
            t_left = -cx / np.cos(angle)
            t_right = (w - cx) / np.cos(angle)
            y_left = cy + t_left * np.sin(angle)
            y_right = cy + t_right * np.sin(angle)
            pt1 = (0, int(round(y_left)))
            pt2 = (w, int(round(y_right)))
        
        cv2.line(overlay, pt1, pt2, colors[c], 2, cv2.LINE_AA)
    
    # 叠加透明图层
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    hva = -1
    ima = -1
    angleLabel = ["HVA: ", "IMA: "]
    
    # 当三个通道的方向都有效时计算角度差
    if len(directions) == 3 and (directions[0] is not None) and (directions[1] is not None) and (directions[2] is not None):
        # 计算通道0和通道1的夹角
        dot01 = np.clip(np.dot(directions[0], directions[1]), -1.0, 1.0)
        angle_diff_01 = np.arccos(dot01) * 180 / np.pi
        hva = angle_diff_01
        text = f"{angleLabel[0]}{hva:.2f}"
        cv2.putText(image, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 计算通道1和通道2的夹角
        dot12 = np.clip(np.dot(directions[1], directions[2]), -1.0, 1.0)
        angle_diff_12 = np.arccos(dot12) * 180 / np.pi
        ima = angle_diff_12
        text = f"{angleLabel[1]}{ima:.2f}"
        cv2.putText(image, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    if hva > 90:
        hva = 180 - 90
    if ima > 90:
        ima = 180 - 90
    return image, hva, ima


def compute_soft_coordinates(prob_map):
    """
    计算前景的软坐标统计，包括加权质心和偏差。
    
    参数：
      prob_map: (H, W) numpy 数组，每个元素表示该像素的前景概率
      
    返回：
      cx, cy: 加权质心的 x, y 坐标
      dx, dy: 每个像素相对于质心的偏差（与 prob_map 同形状）
    """
    import numpy as np
    H, W = prob_map.shape
    xs = np.arange(W)
    ys = np.arange(H)
    xv, yv = np.meshgrid(xs, ys)
    
    sum_p = np.sum(prob_map) + 1e-8
    cx = np.sum(prob_map * xv) / sum_p
    cy = np.sum(prob_map * yv) / sum_p
    
    dx = xv - cx
    dy = yv - cy
    
    return cx, cy, dx, dy

def compute_pca_angle(prob_map):
    """
    基于前景概率图，计算加权协方差矩阵，并利用 PCA 提取主方向，返回对应的角度（弧度）。
    
    参数：
      prob_map: (H, W) numpy 数组，每个元素表示该像素的前景概率
    
    返回：
      angle: 主方向对应的角度，单位为弧度
      cov: 加权协方差矩阵
      eigvals: 协方差矩阵的特征值
      eigvecs: 协方差矩阵的特征向量
    """
    import numpy as np
    cx, cy, dx, dy = compute_soft_coordinates(prob_map)
  
    cov_xx = np.sum(prob_map * (dx ** 2))
    cov_xy = np.sum(prob_map * dx * dy)
    cov_yy = np.sum(prob_map * (dy ** 2))
    
    cov = np.array([[cov_xx, cov_xy],
                    [cov_xy, cov_yy]])
    
    eigvals, eigvecs = np.linalg.eig(cov)
    principal_index = np.argmax(eigvals)
    principal_vector = eigvecs[:, principal_index]
    
    angle = np.arctan2(principal_vector[1], principal_vector[0])
    
    
    return angle, cov, eigvals, eigvecs



def line_regression(image_ori, bbox, heatmap, type=cv2.DIST_WELSCH, constant=0):
    x1, y1, x2, y2 = bbox
    image = image_ori[y1:y2, x1:x2]
    
    # 将 heatmap 转为二值化形式
    heatmap = np.where(heatmap > 0.5, 1, 0).astype(np.uint8)  # heatmap shape: (channels, height, width)
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # 蓝、绿、红
    h, w, _ = image.shape
    n_heatmap = np.zeros([heatmap.shape[0], h, w])  # 通道优先
    
    # 调整每个通道的 heatmap 尺寸
    for c in range(heatmap.shape[0]):
        n_heatmap[c] = cv2.resize(heatmap[c], (w, h))
    
    heatmap = np.array(n_heatmap, np.uint8)
    dxy = []
    angleLabel = ["HVA: ", "IMA: "]
    
    # 创建带透明度的叠加层
    overlay = image.copy()
    
    # 逐通道处理 heatmap
    for c in range(heatmap.shape[0]):  # 通道优先
        pnts = np.where(heatmap[c])
        if len(pnts[0]) == 0:
            continue
        pnts = np.swapaxes(pnts, 0, -1)
        dy, dx, y0, x0 = cv2.fitLine(pnts, type, constant, 0.01, 0.01)
        
        # 绘制线条
        if abs(dy) > 0.01:
            top_x = dx / dy * (-y0) + x0
            bottom_x = dx / dy * (h - y0) + x0
            cv2.line(overlay, (int(top_x), 0), (int(bottom_x), int(h)), colors[c], 2, cv2.LINE_AA)
        else:
            cv2.line(overlay, (0, int(y0)), (int(w), int(y0)), colors[c], 2, cv2.LINE_AA)
        dxy.append(np.array([dx[0], dy[0]]))
    
    # 增加透明度效果
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    hva = -1
    ima = -1
    # 计算并绘制角度
    if len(dxy) == 3:
        for c in range(2):
            angle = np.arccos(np.dot(dxy[c], dxy[c + 1])) * 180 / 3.14
            text = f"{angleLabel[c]}{angle:.2f}"
            text_x, text_y = 30, 40 * c + 40            
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)  # 白色文字
            if c == 0:
                hva = angle
            if c == 1:
                ima = angle
    
    return image, hva, ima

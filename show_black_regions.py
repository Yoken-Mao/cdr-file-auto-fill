import os
import numpy as np
import cv2


def highlight_black_regions(img_path, output_dir='black_regions_output'):
    """
    检测并标出图片中的黑色区域
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    # --- 方法1: 简单阈值检测 ---
    _, black_mask1 = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    black_mask1 = cv2.morphologyEx(black_mask1, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours1, _ = cv2.findContours(black_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制结果1
    img_result1 = img.copy()
    for cnt in contours1:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 30 or bh < 30:
            continue
        # 绘制黑色区域轮廓
        cv2.rectangle(img_result1, (x, y), (x + bw, y + bh), (0, 0, 255), 8)
        # 绘制搜索范围（红色虚线）
        padding = max(bw, bh) * 0.8
        sx1 = max(0, int(x - padding))
        sy1 = max(0, int(y - padding))
        sx2 = min(w, int(x + bw + padding))
        sy2 = min(h, int(y + bh + padding))
        cv2.rectangle(img_result1, (sx1, sy1), (sx2, sy2), (0, 255, 255), 4)

    cv2.putText(img_result1, "Red: Black Region, Yellow: Search Range", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)

    # --- 方法2: 自适应阈值 ---
    black_mask2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 51, 10)
    black_mask2 = cv2.morphologyEx(black_mask2, cv2.MORPH_CLOSE, kernel)

    contours2, _ = cv2.findContours(black_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_result2 = img.copy()
    for cnt in contours2:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 30 or bh < 30:
            continue
        cv2.rectangle(img_result2, (x, y), (x + bw, y + bh), (255, 0, 0), 8)

    cv2.putText(img_result2, "Adaptive Threshold", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 8)

    # --- 保存蒙版 ---
    cv2.imwrite(os.path.join(output_dir, 'black_mask_thresh.png'), black_mask1)
    cv2.imwrite(os.path.join(output_dir, 'black_mask_adaptive.png'), black_mask2)

    # --- 保存结果 ---
    output_path1 = os.path.join(output_dir, 'black_regions_thresh.png')
    output_path2 = os.path.join(output_dir, 'black_regions_adaptive.png')

    # 缩放保存（原图太大）
    scale = 0.25
    img_result1_small = cv2.resize(img_result1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    img_result2_small = cv2.resize(img_result2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path1, img_result1_small)
    cv2.imwrite(output_path2, img_result2_small)

    # 同时保存完整分辨率的裁剪区域 - 缩小裁剪范围
    mid_x = w // 2
    mid_y = h // 2
    crop_size = 1000  # 裁剪范围缩小一半
    cx1 = max(0, mid_x - crop_size)
    cy1 = max(0, mid_y - crop_size)
    cx2 = min(w, mid_x + crop_size)
    cy2 = min(h, mid_y + crop_size)

    # 重新读取原图进行裁剪，确保是原始内容
    img_original = cv2.imread(img_path)
    img_crop = img_original[cy1:cy2, cx1:cx2]

    # 在裁剪图上重新绘制标注
    img_result1_crop = img_crop.copy()
    for cnt in contours1:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 30 or bh < 30:
            continue
        # 检查是否在裁剪范围内
        if x + bw < cx1 or x > cx2 or y + bh < cy1 or y > cy2:
            continue
        # 转换到裁剪图坐标
        rx1 = max(0, x - cx1)
        ry1 = max(0, y - cy1)
        rx2 = min(cx2 - cx1, x + bw - cx1)
        ry2 = min(cy2 - cy1, y + bh - cy1)
        if rx2 > rx1 and ry2 > ry1:
            cv2.rectangle(img_result1_crop, (rx1, ry1), (rx2, ry2), (0, 0, 255), 8)
        # 绘制搜索范围
        padding = max(bw, bh) * 0.8
        sx1 = max(0, int(x - padding) - cx1)
        sy1 = max(0, int(y - padding) - cy1)
        sx2 = min(cx2 - cx1, int(x + bw + padding) - cx1)
        sy2 = min(cy2 - cy1, int(y + bh + padding) - cy1)
        if sx2 > sx1 and sy2 > sy1:
            cv2.rectangle(img_result1_crop, (sx1, sy1), (sx2, sy2), (0, 255, 255), 4)

    cv2.imwrite(os.path.join(output_dir, 'center_crop_original.png'), img_crop)
    cv2.imwrite(os.path.join(output_dir, 'center_crop_with_boxes.png'), img_result1_crop)

    print(f"黑色区域检测结果已保存到: {output_dir}")
    print(f"  - black_regions_thresh.png (阈值检测，缩放25%)")
    print(f"  - black_regions_adaptive.png (自适应阈值，缩放25%)")
    print(f"  - center_crop_*.png (中间区域裁剪)")


if __name__ == '__main__':
    img_path = r'C:/Users/1/PyCharmMiscProject/pictures/20260224-2219-2-S V3.0.png'
    highlight_black_regions(img_path)

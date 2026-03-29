import os
import numpy as np
import cv2


def detect_black_region(img_gray):
    """
    检测图片中的黑色区域，返回边界框 (x, y, w, h)
    """
    # 检测黑色区域 (低亮度区域)
    _, black_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    # 先用小kernel去除噪声
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_small)

    # 再用非常大的kernel连接所有相邻黑色区域，形成一大块
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_large)

    # 查找轮廓
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        return (x, y, w, h), black_mask
    else:
        return None, black_mask


def mask_black_region_with_expand(img, expand_pixels=50, fill_color=(255, 255, 255)):
    """
    检测图片中的黑色区域，外扩指定像素后用白色填充
    返回填充后的图片
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    black_rect, black_mask = detect_black_region(img_gray)

    img_result = img.copy()

    if black_rect:
        bx, by, bw, bh = black_rect
        print(f"找到黑色区域: ({bx}, {by}) -> ({bx+bw}, {by+bh}), 尺寸: {bw}x{bh}")

        # 外扩
        bx_exp = max(0, bx - expand_pixels)
        by_exp = max(0, by - expand_pixels)
        bw_exp = min(w - bx_exp, bw + expand_pixels * 2)
        bh_exp = min(h - by_exp, bh + expand_pixels * 2)

        print(f"外扩后区域: ({bx_exp}, {by_exp}) -> ({bx_exp+bw_exp}, {by_exp+bh_exp}), 尺寸: {bw_exp}x{bh_exp}")

        # 用白色填充
        img_result[by_exp:by_exp+bh_exp, bx_exp:bx_exp+bw_exp] = fill_color
    else:
        print("未找到黑色区域")

    return img_result


def split_and_mask_black_region(img_path, output_dir='non_black_output', expand_pixels=50,
                                 jpeg_quality=85, scale=1.0):
    """
    将图片分割成左右两个，分别检测黑色区域并外扩后用白色填充，输出两个填充后的图片

    jpeg_quality: JPEG压缩质量 (0-100)，越小文件越小
    scale: 图片缩放比例 (0.1-1.0)，小于1时缩小图片
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(img_path)
    if img is None:
        print(f"[FAIL] 无法读取图片: {img_path}")
        return None

    height, width = img.shape[:2]
    mid_x = width // 2

    left_img = img[:, :mid_x]
    right_img = img[:, mid_x:]

    print(f"\n--- 处理左半图片 ---")
    left_masked = mask_black_region_with_expand(left_img, expand_pixels=expand_pixels)
    # 缩放
    if scale < 1.0:
        new_h = int(left_masked.shape[0] * scale)
        new_w = int(left_masked.shape[1] * scale)
        left_masked = cv2.resize(left_masked, (new_w, new_h), interpolation=cv2.INTER_AREA)
    left_output_path = os.path.join(output_dir, 'left_masked.jpg')
    cv2.imwrite(left_output_path, left_masked, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    left_size = os.path.getsize(left_output_path) / 1024
    print(f"左图已保存: {left_output_path} (尺寸: {left_masked.shape[1]}x{left_masked.shape[0]}, 大小: {left_size:.1f} KB)")

    print(f"\n--- 处理右半图片 ---")
    right_masked = mask_black_region_with_expand(right_img, expand_pixels=expand_pixels)
    # 缩放
    if scale < 1.0:
        new_h = int(right_masked.shape[0] * scale)
        new_w = int(right_masked.shape[1] * scale)
        right_masked = cv2.resize(right_masked, (new_w, new_h), interpolation=cv2.INTER_AREA)
    right_output_path = os.path.join(output_dir, 'right_masked.jpg')
    cv2.imwrite(right_output_path, right_masked, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    right_size = os.path.getsize(right_output_path) / 1024
    print(f"右图已保存: {right_output_path} (尺寸: {right_masked.shape[1]}x{right_masked.shape[0]}, 大小: {right_size:.1f} KB)")

    return {
        'left_masked': left_masked,
        'right_masked': right_masked,
        'left_output_path': left_output_path,
        'right_output_path': right_output_path
    }


if __name__ == '__main__':
    img_path = r'C:/Users/1/PyCharmMiscProject/pictures/20260224-2219-2-S V3.0.png'

    print("=" * 60)
    print("分割图片并将黑色区域外扩后用白色填充")
    print("=" * 60)
    # jpeg_quality: JPEG压缩质量 (0-100)，推荐 60-90
    # scale: 缩放比例 (0.1-1.0)，0.5 表示长宽各缩小一半
    split_and_mask_black_region(img_path, expand_pixels=90, jpeg_quality=40, scale=0.8)

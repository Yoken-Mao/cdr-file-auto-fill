import os
from PIL import Image, ImageDraw, ImageFile

# 提高 PIL 的限制
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def find_anchor_points_pil(img, anchor, debug_dir='debug_anchor_pil', debug_prefix='', search_region=None):
    """
    使用 PIL 查找锚定点
    img: PIL Image 对象
    anchor: PIL Image 对象（锚定图片）
    search_region: (x1, y1, x2, y2) 限制搜索区域
    """
    img_gray = img.convert('L')
    anchor_gray = anchor.convert('L')

    img_w, img_h = img_gray.size
    anchor_w, anchor_h = anchor_gray.size

    print(f"锚定图片尺寸: {anchor_w}x{anchor_h}")
    print(f"主图片尺寸: {img_w}x{img_h}")

    # 确定搜索区域
    if search_region:
        x1, y1, x2, y2 = search_region
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
    else:
        x1, y1, x2, y2 = 0, 0, img_w, img_h

    search_w = x2 - x1
    search_h = y2 - y1
    print(f"搜索区域: ({x1}, {y1}) -> ({x2}, {y2}), 尺寸: {search_w}x{search_h}")

    # 获取锚定图片的像素数据
    anchor_pixels = list(anchor_gray.getdata())
    anchor_mean = sum(anchor_pixels) / len(anchor_pixels)

    # 计算锚定图片的方差用于归一化
    anchor_pixels_arr = [(p - anchor_mean) for p in anchor_pixels]
    anchor_norm = (sum(p * p for p in anchor_pixels_arr)) ** 0.5

    if anchor_norm == 0:
        print("[FAIL] 锚定图片是纯色")
        return []

    # 首先进行粗搜索（大步长）
    print("开始粗搜索...")
    step_x_coarse = max(1, anchor_w // 2)
    step_y_coarse = max(1, anchor_h // 2)

    coarse_matches = []

    y = y1
    while y <= y2 - anchor_h:
        x = x1
        while x <= x2 - anchor_w:
            # 快速计算平均亮度差异作为初步筛选
            region = img_gray.crop((x, y, x + anchor_w, y + anchor_h))
            region_pixels = list(region.getdata())
            region_mean = sum(region_pixels) / len(region_pixels)

            # 如果平均亮度差异太大，直接跳过
            if abs(region_mean - anchor_mean) > 50:
                x += step_x_coarse
                continue

            # 计算归一化相关系数
            region_pixels_arr = [(p - region_mean) for p in region_pixels]
            region_norm = (sum(p * p for p in region_pixels_arr)) ** 0.5

            if region_norm == 0:
                x += step_x_coarse
                continue

            numerator = sum(a * b for a, b in zip(anchor_pixels_arr, region_pixels_arr))
            score = numerator / (anchor_norm * region_norm)

            if score > 0.5:  # 粗搜索阈值低一点
                coarse_matches.append((x, y, score))

            x += step_x_coarse
        y += step_y_coarse

    print(f"粗搜索找到 {len(coarse_matches)} 个候选点")

    # 对每个候选点进行精搜索（小步长）
    print("开始精搜索...")
    fine_matches = []
    step_x_fine = max(1, anchor_w // 8)
    step_y_fine = max(1, anchor_h // 8)

    for cx, cy, cscore in coarse_matches:
        # 在候选点周围小范围搜索
        fine_x1 = max(x1, cx - step_x_coarse)
        fine_y1 = max(y1, cy - step_y_coarse)
        fine_x2 = min(x2 - anchor_w, cx + step_x_coarse)
        fine_y2 = min(y2 - anchor_h, cy + step_y_coarse)

        y = fine_y1
        while y <= fine_y2:
            x = fine_x1
            while x <= fine_x2:
                region = img_gray.crop((x, y, x + anchor_w, y + anchor_h))
                region_pixels = list(region.getdata())
                region_mean = sum(region_pixels) / len(region_pixels)

                region_pixels_arr = [(p - region_mean) for p in region_pixels]
                region_norm = (sum(p * p for p in region_pixels_arr)) ** 0.5

                if region_norm == 0:
                    x += step_x_fine
                    continue

                numerator = sum(a * b for a, b in zip(anchor_pixels_arr, region_pixels_arr))
                score = numerator / (anchor_norm * region_norm)

                if score > 0.75:
                    fine_matches.append((x, y, score))

                x += step_x_fine
            y += step_y_fine

    print(f"精搜索找到 {len(fine_matches)} 个匹配点")

    # 非极大值抑制
    filtered = []
    min_distance = max(anchor_w, anchor_h) * 1.2

    # 按分数排序
    for pt in sorted(fine_matches, key=lambda x: x[2], reverse=True):
        x, y, score = pt
        too_close = False
        for fx, fy, _ in filtered:
            if ((x - fx) ** 2 + (y - fy) ** 2) ** 0.5 < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append((x, y, score))

    print(f"过滤后剩余 {len(filtered)} 个匹配点")

    # 排序：从上到下，从左到右
    if filtered:
        sorted_by_y = sorted(filtered, key=lambda p: p[1])
        rows = []
        current_row = [sorted_by_y[0]]
        row_threshold = anchor_h * 0.8

        for p in sorted_by_y[1:]:
            if abs(p[1] - current_row[0][1]) < row_threshold:
                current_row.append(p)
            else:
                rows.append(sorted(current_row, key=lambda x: x[0]))
                current_row = [p]
        rows.append(sorted(current_row, key=lambda x: x[0]))

        final_sorted = []
        for row in rows:
            final_sorted.extend(row)
        filtered = final_sorted

    # 绘制调试图片
    if debug_dir:
        img_debug = img.convert('RGB')
        draw = ImageDraw.Draw(img_debug)

        for i, (x, y, score) in enumerate(filtered):
            # 画矩形
            draw.rectangle([x, y, x + anchor_w, y + anchor_h], outline='red', width=10)
            # 画中心点
            cx, cy = x + anchor_w // 2, y + anchor_h // 2
            draw.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill='green')
            # 标注序号
            draw.text((x, y - 60), f"#{i+1}\n({score:.2f})", fill='red', font_size=50)

        debug_path = os.path.join(debug_dir, f'{debug_prefix}anchor_detection.png')
        img_debug.save(debug_path)
        print(f"[DEBUG] 调试图片已保存: {debug_path}")

    return [(x + anchor_w // 2, y + anchor_h // 2, score) for x, y, score in filtered]


def split_image_and_find_anchors_pil(img_path, anchor_path, debug_dir='debug_anchor_pil'):
    """
    分割图片为左右两半，分别寻找锚定点
    """
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    img = Image.open(img_path)
    anchor = Image.open(anchor_path)

    width, height = img.size
    mid_x = width // 2

    # 保存左右两半
    left_img = img.crop((0, 0, mid_x, height))
    right_img = img.crop((mid_x, 0, width, height))

    left_path = os.path.join(debug_dir, 'left_half.png')
    right_path = os.path.join(debug_dir, 'right_half.png')
    left_img.save(left_path)
    right_img.save(right_path)

    print(f"\n--- 处理左半图片 ---")
    # 左半图片：重点搜索右半部分
    left_search_region = (mid_x // 2, 0, mid_x, height)
    left_anchors = find_anchor_points_pil(left_img, anchor, debug_dir, 'left_', left_search_region)

    print(f"\n--- 处理右半图片 ---")
    # 右半图片：重点搜索左半部分
    right_search_region = (0, 0, (width - mid_x) // 2, height)
    right_anchors = find_anchor_points_pil(right_img, anchor, debug_dir, 'right_', right_search_region)

    # 右半图片的坐标需要加上 mid_x
    right_anchors_shifted = [(x + mid_x, y, score) for x, y, score in right_anchors]

    return {
        'left': left_anchors,
        'right': right_anchors,
        'right_shifted': right_anchors_shifted,
        'mid_x': mid_x
    }


if __name__ == '__main__':
    img_path = r'C:/Users/1/PyCharmMiscProject/pictures/20260224-2219-2-S V3.0.png'
    anchor_path = r'C:/Users/1/PyCharmMiscProject/pictures/base.png'

    results = split_image_and_find_anchors_pil(img_path, anchor_path)

    print("\n=== 最终结果 ===")
    print(f"左半图片锚定点: {len(results['left'])} 个")
    for i, (x, y, score) in enumerate(results['left']):
        print(f"  #{i+1}: ({x}, {y}), score={score:.2f}")

    print(f"\n右半图片锚定点: {len(results['right'])} 个")
    for i, (x, y, score) in enumerate(results['right']):
        print(f"  #{i+1}: ({x}, {y}), score={score:.2f}")

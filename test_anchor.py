import os
import numpy as np
import cv2


def find_black_regions(img_gray, img_color, debug_dir=None, debug_prefix=''):
    """
    找到图片中的黑色区域，并返回黑色区域四个角的搜索范围
    """
    h, w = img_gray.shape

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

    search_regions = []
    black_rect = None

    if len(contours) > 0:
        # 找到最大的那个黑色区域
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest_cnt)
        black_rect = (x, y, bw, bh)

        # 在四个角周围定义搜索区域
        corner_size = int(max(bw, bh) * 0.35)  # 每个角搜索区域大小

        # 左上角
        tl_x1 = max(0, int(x - corner_size))
        tl_y1 = max(0, int(y - corner_size))
        tl_x2 = min(w, int(x + bw * 0.35))
        tl_y2 = min(h, int(y + bh * 0.35))
        search_regions.append((tl_x1, tl_y1, tl_x2, tl_y2, 'top-left'))

        # 右上角
        tr_x1 = max(0, int(x + bw * 0.65))
        tr_y1 = max(0, int(y - corner_size))
        tr_x2 = min(w, int(x + bw + corner_size))
        tr_y2 = min(h, int(y + bh * 0.35))
        search_regions.append((tr_x1, tr_y1, tr_x2, tr_y2, 'top-right'))

        # 左下角
        bl_x1 = max(0, int(x - corner_size))
        bl_y1 = max(0, int(y + bh * 0.65))
        bl_x2 = min(w, int(x + bw * 0.35))
        bl_y2 = min(h, int(y + bh + corner_size))
        search_regions.append((bl_x1, bl_y1, bl_x2, bl_y2, 'bottom-left'))

        # 右下角
        br_x1 = max(0, int(x + bw * 0.65))
        br_y1 = max(0, int(y + bh * 0.65))
        br_x2 = min(w, int(x + bw + corner_size))
        br_y2 = min(h, int(y + bh + corner_size))
        search_regions.append((br_x1, br_y1, br_x2, br_y2, 'bottom-right'))

        print(f"找到主黑色区域: ({x}, {y}) -> ({x+bw}, {y+bh}), 尺寸: {bw}x{bh}")
        for sr in search_regions:
            print(f"  {sr[4]}: ({sr[0]}, {sr[1]}) -> ({sr[2]}, {sr[3]})")
    else:
        # 如果没有找到黑色区域，使用默认搜索区域
        print("未找到明显的黑色区域，使用默认搜索区域")
        search_regions.append((w//5, h//5, w*4//5, h*4//5, 'default'))

    # 保存调试图片
    if debug_dir is not None:
        # 先裁剪中间区域再画框，避免内存不足
        h_full, w_full = img_color.shape[:2]
        if black_rect is not None:
            bx, by, bbw, bbh = black_rect
            crop_margin = max(bbw, bbh) // 2
            cx1 = max(0, bx - crop_margin)
            cy1 = max(0, by - crop_margin)
            cx2 = min(w_full, bx + bbw + crop_margin)
            cy2 = min(h_full, by + bbh + crop_margin)
        else:
            cx1, cy1, cx2, cy2 = 0, 0, min(w_full, 3000), min(h_full, 3000)

        debug_img = img_color[cy1:cy2, cx1:cx2].copy()

        # 画黑色区域框
        if black_rect is not None:
            bx, by, bbw, bbh = black_rect
            rbx1 = max(0, bx - cx1)
            rby1 = max(0, by - cy1)
            rbx2 = min(cx2 - cx1, bx + bbw - cx1)
            rby2 = min(cy2 - cy1, by + bbh - cy1)
            cv2.rectangle(debug_img, (rbx1, rby1), (rbx2, rby2), (0, 0, 255), 8)

        # 画四个角的搜索区域
        colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, (x1, y1, x2, y2, name) in enumerate(search_regions):
            rx1 = max(0, x1 - cx1)
            ry1 = max(0, y1 - cy1)
            rx2 = min(cx2 - cx1, x2 - cx1)
            ry2 = min(cy2 - cy1, y2 - cy1)
            if rx2 > rx1 and ry2 > ry1:
                color = colors[i % len(colors)]
                cv2.rectangle(debug_img, (rx1, ry1), (rx2, ry2), color, 6)
                cv2.putText(debug_img, name, (rx1 + 10, ry1 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        debug_path = os.path.join(debug_dir, f'{debug_prefix}black_regions.png')
        cv2.imwrite(debug_path, debug_img)
        print(f"黑色区域调试图已保存: {debug_path}")

        # 保存mask的裁剪版本
        mask_path = os.path.join(debug_dir, f'{debug_prefix}black_mask.png')
        cv2.imwrite(mask_path, black_mask[cy1:cy2, cx1:cx2])

    # 返回搜索区域（去掉name字段）
    return [(sr[0], sr[1], sr[2], sr[3]) for sr in search_regions]


def find_anchor_points(img_path, anchor_path, debug_dir='debug_anchor', debug_prefix=''):
    """
    在图片中寻找锚定点 - 只在黑色区域四个角搜索
    """
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # 读取主图片和锚定图片
    img = cv2.imread(img_path)
    anchor = cv2.imread(anchor_path)

    if img is None:
        print(f"[FAIL] 无法读取图片: {img_path}")
        return []
    if anchor is None:
        print(f"[FAIL] 无法读取锚定图片: {anchor_path}")
        return []

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    anchor_gray = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)

    h, w = anchor_gray.shape
    img_h, img_w = img_gray.shape
    print(f"锚定图片尺寸: {w}x{h}")
    print(f"主图片尺寸: {img_w}x{img_h}")

    # 找到黑色区域四个角的搜索范围
    search_regions = find_black_regions(img_gray, img, debug_dir, debug_prefix)
    print(f"搜索区域数量: {len(search_regions)}")

    all_candidates = []

    # 使用较低的初始阈值，收集更多候选
    methods = [
        ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED, 0.82),
        ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED, 0.85),
    ]

    # 尺度范围
    scales = [1.0, 0.97, 0.94, 0.91, 0.88, 0.85,
              1.03, 1.06, 1.09, 1.12, 1.15]

    # 在每个搜索区域中搜索
    for region_idx, (search_x1, search_y1, search_x2, search_y2) in enumerate(search_regions):
        print(f"处理搜索区域 {region_idx + 1}: ({search_x1}, {search_y1}) -> ({search_x2}, {search_y2})")

        search_img = img_gray[search_y1:search_y2, search_x1:search_x2]

        if search_img.shape[0] < h or search_img.shape[1] < w:
            print("  搜索区域太小，跳过")
            continue

        for method_name, method, threshold in methods:
            for scale in scales:
                if scale == 1.0:
                    scaled_anchor = anchor_gray
                else:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    if new_w < 10 or new_h < 10:
                        continue
                    scaled_anchor = cv2.resize(anchor_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

                sh, sw = scaled_anchor.shape
                if sh > search_img.shape[0] or sw > search_img.shape[1]:
                    continue

                # 进行模板匹配
                try:
                    result = cv2.matchTemplate(search_img, scaled_anchor, method)
                except Exception as e:
                    continue

                # 获取高分匹配点
                if method == cv2.TM_SQDIFF_NORMED:
                    locations = np.where(result <= threshold)
                else:
                    locations = np.where(result >= threshold)

                if len(locations[0]) > 0:
                    scores = result[locations[0], locations[1]]
                    # 只保留前30个高分点
                    if len(scores) > 30:
                        if method == cv2.TM_SQDIFF_NORMED:
                            top_indices = np.argsort(scores)[:30]
                        else:
                            top_indices = np.argsort(scores)[::-1][:30]
                        y_coords = locations[0][top_indices]
                        x_coords = locations[1][top_indices]
                        scores = scores[top_indices]
                    else:
                        y_coords = locations[0]
                        x_coords = locations[1]

                    for i in range(len(x_coords)):
                        x = x_coords[i]
                        y = y_coords[i]
                        score = scores[i]
                        if method == cv2.TM_SQDIFF_NORMED:
                            score = 1.0 - score

                        # 转换到原图坐标
                        orig_x = x + search_x1
                        orig_y = y + search_y1

                        all_candidates.append({
                            'x': orig_x,
                            'y': orig_y,
                            'score': score,
                            'sw': sw,
                            'sh': sh,
                            'scale': scale,
                            'region': region_idx
                        })

    print(f"找到 {len(all_candidates)} 个初步候选点")

    if len(all_candidates) == 0:
        # 保存空的调试图片
        debug_path = os.path.join(debug_dir, f'{debug_prefix}anchor_detection.png')
        cv2.imwrite(debug_path, img[:min(3000, img.shape[0]), :min(3000, img.shape[1])])
        return []

    # ========== 每个搜索区域保留最好的结果 ==========
    best_per_region = {}
    for cand in all_candidates:
        region = cand['region']
        if region not in best_per_region or cand['score'] > best_per_region[region]['score']:
            best_per_region[region] = cand

    filtered = list(best_per_region.values())
    filtered.sort(key=lambda x: x['score'], reverse=True)

    print(f"每个区域取最好的，剩余 {len(filtered)} 个匹配点")

    # ========== 绘制调试图片 ==========
    result_list = []

    # 确定裁剪范围
    h_full, w_full = img.shape[:2]
    if len(filtered) > 0:
        all_x = [p['x'] for p in filtered]
        all_y = [p['y'] for p in filtered]
        cx1 = max(0, min(all_x) - 500)
        cy1 = max(0, min(all_y) - 500)
        cx2 = min(w_full, max(all_x) + max(p['sw'] for p in filtered) + 500)
        cy2 = min(h_full, max(all_y) + max(p['sh'] for p in filtered) + 500)
    elif len(search_regions) > 0:
        all_sx = []
        all_sy = []
        for sr in search_regions:
            all_sx.extend([sr[0], sr[2]])
            all_sy.extend([sr[1], sr[3]])
        cx1 = max(0, min(all_sx) - 200)
        cy1 = max(0, min(all_sy) - 200)
        cx2 = min(w_full, max(all_sx) + 200)
        cy2 = min(h_full, max(all_sy) + 200)
    else:
        cx1, cy1, cx2, cy2 = 0, 0, min(w_full, 3000), min(h_full, 3000)

    img_debug = img[cy1:cy2, cx1:cx2].copy()

    corner_names = ['TL', 'TR', 'BL', 'BR']
    for i, pt in enumerate(filtered):
        x, y = pt['x'], pt['y']
        sw, sh = pt['sw'], pt['sh']
        score = pt['score']
        scale = pt['scale']
        region = pt['region']

        # 转换到裁剪后的坐标
        rx = x - cx1
        ry = y - cy1

        if rx >= 0 and ry >= 0 and rx + sw < img_debug.shape[1] and ry + sh < img_debug.shape[0]:
            cv2.rectangle(img_debug, (rx, ry), (rx + sw, ry + sh), (0, 0, 255), 10)
            rcx, rcy = rx + sw // 2, ry + sh // 2
            cv2.circle(img_debug, (rcx, rcy), 20, (0, 255, 0), -1)
            name = corner_names[region] if region < len(corner_names) else str(region)
            label = f"#{i+1} {name}\n({score:.3f})\n{scale:.2f}"
            cv2.putText(img_debug, label, (rx, ry - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        cx, cy = x + sw // 2, y + sh // 2
        result_list.append((cx, cy, score))

    debug_path = os.path.join(debug_dir, f'{debug_prefix}anchor_detection.png')
    cv2.imwrite(debug_path, img_debug)
    print(f"[DEBUG] 调试图片已保存: {debug_path}")

    return result_list


def split_image_and_find_anchors(img_path, anchor_path, debug_dir='debug_anchor'):
    """
    分割图片为左右两半，分别寻找锚定点
    """
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    img = cv2.imread(img_path)
    if img is None:
        print(f"[FAIL] 无法读取图片: {img_path}")
        return None

    height, width = img.shape[:2]
    mid_x = width // 2

    left_img = img[:, :mid_x]
    right_img = img[:, mid_x:]

    left_path = os.path.join(debug_dir, 'left_half.png')
    right_path = os.path.join(debug_dir, 'right_half.png')
    cv2.imwrite(left_path, left_img)
    cv2.imwrite(right_path, right_img)

    print(f"\n--- 处理左半图片 ---")
    left_anchors = find_anchor_points(left_path, anchor_path, debug_dir, 'left_')

    print(f"\n--- 处理右半图片 ---")
    right_anchors = find_anchor_points(right_path, anchor_path, debug_dir, 'right_')

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

    results = split_image_and_find_anchors(img_path, anchor_path)

    if results:
        print("\n=== 最终结果 ===")
        print(f"左半图片锚定点: {len(results['left'])} 个")
        for i, (x, y, score) in enumerate(results['left']):
            print(f"  #{i+1}: ({x}, {y}), score={score:.3f}")

        print(f"\n右半图片锚定点: {len(results['right'])} 个")
        for i, (x, y, score) in enumerate(results['right']):
            print(f"  #{i+1}: ({x}, {y}), score={score:.3f}")

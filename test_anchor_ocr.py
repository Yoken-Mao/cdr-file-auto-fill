import os
import sys
from PIL import Image
import pytesseract
import numpy as np
import cv2

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))
ANCHOR_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pictures', 'base.png')


def find_anchor_points_opencv(img_gray, anchor_gray, is_left=True):
    """
    使用OpenCV在图片中寻找锚定点
    返回按从上到下、从左到右排序的锚定点中心点坐标 [(x,y), ...]
    """
    h, w = anchor_gray.shape
    img_h, img_w = img_gray.shape

    # 定义搜索区域：中间区域
    search_x1 = max(0, img_w // 4)
    search_x2 = min(img_w, img_w * 3 // 4)
    search_y1 = max(0, img_h // 4)
    search_y2 = min(img_h, img_h * 3 // 4)

    search_img = img_gray[search_y1:search_y2, search_x1:search_x2]

    all_matches = []

    # 使用多种匹配方法
    methods = [
        (cv2.TM_CCOEFF_NORMED, 0.5),
        (cv2.TM_CCORR_NORMED, 0.8),
    ]

    # 多尺度搜索
    scales = [1.0, 0.9, 0.8, 0.7, 1.1, 1.2, 1.3]

    for method, threshold in methods:
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
            except Exception:
                continue

            # 获取高分匹配点
            if method == cv2.TM_SQDIFF_NORMED:
                locations = np.where(result <= threshold)
            else:
                locations = np.where(result >= threshold)

            if len(locations[0]) > 0:
                scores = result[locations[0], locations[1]]
                # 只保留前50个
                if len(scores) > 50:
                    if method == cv2.TM_SQDIFF_NORMED:
                        top_indices = np.argsort(scores)[:50]
                    else:
                        top_indices = np.argsort(scores)[::-1][:50]
                    y_coords = locations[0][top_indices]
                    x_coords = locations[1][top_indices]
                    scores = scores[top_indices]
                else:
                    y_coords = locations[0]
                    x_coords = locations[1]

                for i in range(len(x_coords)):
                    x = x_coords[i] + search_x1
                    y = y_coords[i] + search_y1
                    score = scores[i]
                    if method == cv2.TM_SQDIFF_NORMED:
                        score = 1.0 - score
                    all_matches.append((x, y, score, sw, sh))

    # 非极大值抑制
    filtered = []
    min_distance = max(w, h) * 1.5

    for pt in sorted(all_matches, key=lambda x: x[2], reverse=True):
        x, y, score, sw, sh = pt
        too_close = False
        for fx, fy, _, _, _ in filtered:
            if ((x - fx) ** 2 + (y - fy) ** 2) ** 0.5 < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append((x, y, score, sw, sh))
        if len(filtered) >= 10:
            break

    # 排序：按y坐标分上行和下行，每行内按x坐标排序
    centers = []
    if filtered:
        # 按y坐标排序
        sorted_by_y = sorted(filtered, key=lambda p: p[1])
        # 分成上下两行（根据中间y值分割）
        if len(sorted_by_y) >= 2:
            mid_y = (sorted_by_y[0][1] + sorted_by_y[-1][1]) / 2
            top_row = [p for p in sorted_by_y if p[1] < mid_y]
            bottom_row = [p for p in sorted_by_y if p[1] >= mid_y]
            # 每行按x坐标排序
            top_row_sorted = sorted(top_row, key=lambda p: p[0])
            bottom_row_sorted = sorted(bottom_row, key=lambda p: p[0])
            # 合并：先上后下
            final_sorted = top_row_sorted + bottom_row_sorted
        else:
            final_sorted = sorted_by_y

        # 返回中心点坐标
        centers = [(x + sw // 2, y + sh // 2) for x, y, score, sw, sh in final_sorted]

    return centers


def ocr_single_region(region_img):
    """对单个区域进行OCR识别"""
    from PIL import ImageEnhance

    region_w, region_h = region_img.size

    all_results = []
    scales = [1.0]
    if region_w > 4000 or region_h > 4000:
        scales.extend([0.5, 0.3, 0.25])

    for scale in scales:
        if scale == 1.0:
            scaled_img = region_img
        else:
            new_w = int(region_w * scale)
            new_h = int(region_h * scale)
            scaled_img = region_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        all_results.append(scaled_img)

    best_text = ""
    best_score = 0

    for region_img_var in all_results:
        img_gray = region_img_var.convert('L')

        img_variants = [
            ('原始灰度', img_gray),
            ('对比度x2', ImageEnhance.Contrast(img_gray).enhance(2.0)),
            ('对比度x3', ImageEnhance.Contrast(img_gray).enhance(3.0)),
        ]

        for threshold in [150, 180, 200]:
            img_variants.append((f'二值化{threshold}', img_gray.point(lambda x: 0 if x < threshold else 255, '1')))

        test_configs = [
            ('中英文-PSM6', 'chi_sim+eng', r'--oem 3 --psm 6'),
            ('纯英文-PSM6', 'eng', r'--oem 3 --psm 6'),
            ('中英文-PSM3', 'chi_sim+eng', r'--oem 3 --psm 3'),
        ]

        for img_name, img_var in img_variants:
            for cfg_name, lang, cfg in test_configs:
                try:
                    text = pytesseract.image_to_string(img_var, lang=lang, config=cfg)
                    text = text.strip()

                    if text:
                        score = len(text)
                        has_digits = any(c.isdigit() for c in text)
                        has_multiply = '*' in text or '×' in text or 'x' in text or 'X' in text
                        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)

                        if has_digits:
                            score += 20
                        if has_chinese:
                            score += 20
                        if has_multiply:
                            score += 40

                        if score > best_score:
                            best_score = score
                            best_text = text
                except Exception:
                    continue

    return best_text


def test_anchor_ocr():
    """测试基于锚定点的裁剪和OCR识别"""
    img_path = r'C:/Users/1/PyCharmMiscProject/pictures/20260224-2219-2-S V3.0.png'
    debug_dir = 'test_anchor_ocr'

    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    print("=" * 60)
    print("测试基于锚定点的裁剪和OCR识别")
    print("=" * 60)

    # 读取图片
    img = Image.open(img_path)
    width, height = img.size
    print(f"图片尺寸: {img.size}")

    mid_x = width // 2

    # 分割为左右两半
    left_img = img.crop((0, 0, mid_x, height))
    right_img = img.crop((mid_x, 0, width, height))

    # 加载锚定图片
    anchor_img = cv2.imread(ANCHOR_IMAGE_PATH)
    anchor_gray = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2GRAY)

    # 转换为OpenCV格式
    left_img_cv = cv2.cvtColor(np.array(left_img), cv2.COLOR_RGB2GRAY)
    right_img_cv = cv2.cvtColor(np.array(right_img), cv2.COLOR_RGB2GRAY)

    # 寻找锚定点
    print("\n--- 寻找锚定点 ---")
    left_anchors = find_anchor_points_opencv(left_img_cv, anchor_gray, is_left=True)
    right_anchors = find_anchor_points_opencv(right_img_cv, anchor_gray, is_left=False)
    print(f"左半图片找到 {len(left_anchors)} 个锚定点")
    print(f"右半图片找到 {len(right_anchors)} 个锚定点")

    # 根据锚定点裁剪表格
    left_table = None
    right_table = None

    if len(left_anchors) >= 4:
        num_anchors = len(left_anchors)
        anchor_bottom_left = left_anchors[num_anchors - 2]
        anchor_bottom_right = left_anchors[num_anchors - 1]

        crop_x1 = max(0, anchor_bottom_left[0] - 200)
        crop_y1 = max(anchor_bottom_left[1], anchor_bottom_right[1]) + 50
        crop_x2 = mid_x
        crop_y2 = height

        if crop_x2 - crop_x1 > 100 and crop_y2 - crop_y1 > 100:
            left_table = left_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            print(f"\n左半表格裁剪区域: ({crop_x1}, {crop_y1}) -> ({crop_x2}, {crop_y2})")

    if len(right_anchors) >= 4:
        num_anchors = len(right_anchors)
        anchor_bottom_left = right_anchors[num_anchors - 2]
        anchor_bottom_right = right_anchors[num_anchors - 1]

        crop_x1 = 0
        crop_y1 = max(anchor_bottom_left[1], anchor_bottom_right[1]) + 50
        crop_x2 = anchor_bottom_right[0] + 200
        crop_y2 = height

        if crop_x2 - crop_x1 > 100 and crop_y2 - crop_y1 > 100:
            right_table = right_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            print(f"右半表格裁剪区域: ({crop_x1}, {crop_y1}) -> ({crop_x2}, {crop_y2})")

    # 保存裁剪后的表格
    if left_table:
        left_table.save(os.path.join(debug_dir, 'left_table.png'))
        print(f"左半表格已保存, 尺寸: {left_table.size}")

    if right_table:
        right_table.save(os.path.join(debug_dir, 'right_table.png'))
        print(f"右半表格已保存, 尺寸: {right_table.size}")

    # 进行OCR识别
    print("\n--- OCR识别 ---")
    if left_table:
        print("识别左半表格...")
        left_text = ocr_single_region(left_table)
        print(f"左半表格识别结果:\n{left_text}\n")
    else:
        left_text = ""

    if right_table:
        print("识别右半表格...")
        right_text = ocr_single_region(right_table)
        print(f"右半表格识别结果:\n{right_text}\n")
    else:
        right_text = ""

    # 合并结果
    all_text_parts = []
    if left_text:
        all_text_parts.append("【左半表格】\n" + left_text)
    if right_text:
        all_text_parts.append("【右半表格】\n" + right_text)

    final_text = "\n---\n".join(all_text_parts) if all_text_parts else ""

    print("=" * 60)
    print("最终合并结果:")
    print(final_text)
    print("=" * 60)

    # 保存结果
    with open(os.path.join(debug_dir, 'ocr_result.txt'), 'w', encoding='utf-8') as f:
        f.write(final_text)
    print(f"\n结果已保存到: {debug_dir}")


if __name__ == '__main__':
    test_anchor_ocr()

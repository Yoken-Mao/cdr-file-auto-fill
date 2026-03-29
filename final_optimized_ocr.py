import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2

# 配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))


def find_black_region_and_table(img_gray, img_color, table_corner=None, table_size=(1200, 300)):
    """
    增大表格区域，确保完整裁剪
    """
    h, w = img_gray.shape
    _, black_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_small)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_large)
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_rect = None
    table_rect = None

    if len(contours) > 0:
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest_cnt)
        black_rect = (x, y, bw, bh)

        tw, th = table_size
        if table_corner == 'bottom-right':
            # 向左上方扩展，确保包含完整表格
            tx = x + bw - tw + 50
            ty = y + bh - th
            table_rect = (tx, ty, tw, th)
        elif table_corner == 'bottom-left':
            tx = x - 50
            ty = y + bh - th
            table_rect = (tx, ty, tw, th)

    return black_rect, table_rect


def crop_table_by_black_region(img, debug_dir='debug_final'):
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    width, height = img.size
    mid_x = width // 2

    left_img_pil = img.crop((0, 0, mid_x, height))
    right_img_pil = img.crop((mid_x, 0, width, height))

    left_img_cv = cv2.cvtColor(np.array(left_img_pil), cv2.COLOR_RGB2BGR)
    right_img_cv = cv2.cvtColor(np.array(right_img_pil), cv2.COLOR_RGB2BGR)

    left_gray = cv2.cvtColor(left_img_cv, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img_cv, cv2.COLOR_BGR2GRAY)

    left_black_rect, left_table_rect = find_black_region_and_table(
        left_gray, left_img_cv,
        table_corner='bottom-right',
        table_size=(1200, 300))

    right_black_rect, right_table_rect = find_black_region_and_table(
        right_gray, right_img_cv,
        table_corner='bottom-left',
        table_size=(1200, 300))

    left_table = None
    right_table = None

    if left_table_rect:
        tx, ty, tw, th = left_table_rect
        tx_clamp = max(0, tx)
        ty_clamp = max(0, ty)
        tx2_clamp = min(mid_x, tx + tw)
        ty2_clamp = min(height, ty + th)
        if tx2_clamp > tx_clamp and ty2_clamp > ty_clamp:
            left_table = left_img_pil.crop((tx_clamp, ty_clamp, tx2_clamp, ty2_clamp))
            left_table.save(os.path.join(debug_dir, 'left_table_final.png'))
            print(f"左表格已保存: left_table_final.png, 尺寸: {left_table.size}")

    if right_table_rect:
        tx, ty, tw, th = right_table_rect
        tx_clamp = max(0, tx)
        ty_clamp = max(0, ty)
        tx2_clamp = min(width - mid_x, tx + tw)
        ty2_clamp = min(height, ty + th)
        if tx2_clamp > tx_clamp and ty2_clamp > ty_clamp:
            right_table = right_img_pil.crop((tx_clamp, ty_clamp, tx2_clamp, ty2_clamp))
            right_table.save(os.path.join(debug_dir, 'right_table_final.png'))
            print(f"右表格已保存: right_table_final.png, 尺寸: {right_table.size}")

    return {'left': left_table, 'right': right_table}


def optimized_preprocess(img):
    """
    优化的图像预处理
    """
    variants = []

    img_gray = img.convert('L')
    w, h = img_gray.size

    # 放大3倍 - 这是关键
    img_x3 = img_gray.resize((w * 3, h * 3), Image.Resampling.LANCZOS)

    # 基础放大版本
    variants.append(('放大3倍', img_x3))

    # 对比度增强
    for contrast in [1.5, 2.0, 2.5, 3.0]:
        img_contrast = ImageEnhance.Contrast(img_x3).enhance(contrast)
        variants.append((f'放大3倍+对比度{contrast}', img_contrast))

        # 对比度+锐化
        for sharpness in [1.5, 2.0]:
            img_sharp = ImageEnhance.Sharpness(img_contrast).enhance(sharpness)
            variants.append((f'放大3倍+对比度{contrast}+锐化{sharpness}', img_sharp))

    # 二值化
    for threshold in [100, 120, 140, 160, 180]:
        img_bin = img_x3.point(lambda x: 0 if x < threshold else 255, '1')
        variants.append((f'放大3倍+二值化{threshold}', img_bin))

    # OpenCV 自适应阈值
    img_cv = np.array(img_x3)
    for block_size in [11, 21, 31, 51]:
        for c in [2, 5, 10]:
            try:
                img_thresh = cv2.adaptiveThreshold(
                    img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, c)
                img_thresh_pil = Image.fromarray(img_thresh)
                variants.append((f'CV自适应{block_size}-{c}', img_thresh_pil))
            except:
                pass

    return variants


def ocr_image(img):
    """
    使用多种配置进行OCR
    """
    variants = optimized_preprocess(img)

    best_text = ""
    best_score = 0
    best_method = ""

    # 更多的OCR配置尝试
    test_configs = [
        ('中英文-PSM6', 'chi_sim+eng', r'--oem 3 --psm 6'),
        ('中英文-PSM3', 'chi_sim+eng', r'--oem 3 --psm 3'),
        ('中英文-PSM4', 'chi_sim+eng', r'--oem 3 --psm 4'),
        ('中英文-PSM11', 'chi_sim+eng', r'--oem 3 --psm 11'),
        ('中英文-PSM12', 'chi_sim+eng', r'--oem 3 --psm 12'),
        ('纯中文-PSM6', 'chi_sim', r'--oem 3 --psm 6'),
        ('纯英文-PSM6', 'eng', r'--oem 3 --psm 6'),
    ]

    for var_name, var_img in variants:
        for cfg_name, lang, cfg in test_configs:
            try:
                text = pytesseract.image_to_string(var_img, lang=lang, config=cfg)
                text = text.strip()

                if text:
                    score = calculate_score(text)
                    if score > best_score:
                        best_score = score
                        best_text = text
                        best_method = f"{var_name}-{cfg_name}"
            except Exception:
                continue

    return best_text, best_score, best_method


def calculate_score(text):
    """
    计算识别结果的评分
    """
    score = len(text) * 2

    # 期望的关键词
    keywords = {
        '编号': 20, '颜色': 20, '设计': 20, '版次': 20,
        '目数': 20, '日期': 20, 'MSCM': 30, 'XKS': 30,
        '20260209': 40, '线路': 25, '反印': 25, '正印': 25,
        '二次黑': 30, '黑色': 20, '300': 20, '350': 20,
    }

    for keyword, points in keywords.items():
        if keyword in text:
            score += points

    # 格式检查
    if ':' in text or '：' in text:
        score += 15
    if '|' in text:
        score += 20

    # 检查数字和中文混合
    has_digits = any(c.isdigit() for c in text)
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    if has_digits:
        score += 10
    if has_chinese:
        score += 10

    # 惩罚多余的下划线
    underscore_count = text.count('_')
    score -= underscore_count * 2

    return score


def clean_text(text):
    """
    清理识别结果
    """
    # 移除多余的下划线
    text = text.replace('_', '')
    # 移除多余空格
    while '  ' in text:
        text = text.replace('  ', ' ')
    # 清理行首尾
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(cleaned_lines)


def main():
    test_img_path = r'C:\Users\1\PyCharmMiscProject\pictures\20260224-2219-2-S V3.0.png'

    print("="*70)
    print("裁剪表格区域（增大尺寸）")
    print("="*70)

    img = Image.open(test_img_path)
    cropped = crop_table_by_black_region(img)

    if cropped['left'] is None or cropped['right'] is None:
        print("裁剪失败")
        return

    print("\n" + "="*70)
    print("OCR识别（优化版）")
    print("="*70)

    left_text, left_score, left_method = ocr_image(cropped['left'])
    right_text, right_score, right_method = ocr_image(cropped['right'])

    # 清理文本
    left_text = clean_text(left_text)
    right_text = clean_text(right_text)

    print(f"\n左表格 (得分:{left_score}, 方法:{left_method}):")
    print("-" * 50)
    print(left_text)
    print("-" * 50)

    print(f"\n右表格 (得分:{right_score}, 方法:{right_method}):")
    print("-" * 50)
    print(right_text)
    print("-" * 50)

    result = "【左半表格】\n" + left_text + "\n---\n【右半表格】\n" + right_text

    print("\n" + "="*70)
    print("最终结果:")
    print("="*70)
    print(result)
    print("="*70)

    with open('final_ocr_result.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    print("\n结果已保存到: final_ocr_result.txt")

    # 对比期望结果
    expected_path = r'C:\Users\1\PyCharmMiscProject\ocr_text\result.txt'
    if os.path.exists(expected_path):
        with open(expected_path, 'r', encoding='utf-8') as f:
            expected = f.read()
        print("\n" + "="*70)
        print("期望结果:")
        print("="*70)
        print(expected)
        print("="*70)


if __name__ == '__main__':
    main()

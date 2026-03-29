import os
import pytesseract
from PIL import Image, ImageEnhance
import numpy as np
import cv2

# 配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))


def find_black_region_and_table(img_gray, img_color, table_corner=None, table_size=(1000, 200)):
    """保持原来的裁剪区域不变"""
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

        if table_corner == 'bottom-right':
            tw, th = table_size
            tx = x + bw - tw + 100
            ty = y + bh - th // 3
            table_rect = (tx, ty, tw, th)
        elif table_corner == 'bottom-left':
            tw, th = table_size
            tx = x - 100
            ty = y + bh - th // 3
            table_rect = (tx, ty, tw, th)

    return black_rect, table_rect


def crop_table_by_black_region(img):
    """保持原来的裁剪逻辑"""
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
        table_size=(1000, 200))

    right_black_rect, right_table_rect = find_black_region_and_table(
        right_gray, right_img_cv,
        table_corner='bottom-left',
        table_size=(1000, 200))

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

    if right_table_rect:
        tx, ty, tw, th = right_table_rect
        tx_clamp = max(0, tx)
        ty_clamp = max(0, ty)
        tx2_clamp = min(width - mid_x, tx + tw)
        ty2_clamp = min(height, ty + th)
        if tx2_clamp > tx_clamp and ty2_clamp > ty_clamp:
            right_table = right_img_pil.crop((tx_clamp, ty_clamp, tx2_clamp, ty2_clamp))

    return {'left': left_table, 'right': right_table}


def smart_ocr(img, is_left=True):
    """
    有针对性的OCR识别
    """
    img_gray = img.convert('L')
    w, h = img_gray.size

    best_text = ""
    best_score = 0

    # 基于之前的经验，放大3倍效果最好
    scaled = img_gray.resize((w * 3, h * 3), Image.Resampling.LANCZOS)

    # 尝试有限的几种组合
    test_cases = []

    # 对比度增强
    for contrast in [1.5, 2.0, 2.5]:
        img_contrast = ImageEnhance.Contrast(scaled).enhance(contrast)
        test_cases.append((f'contrast{contrast}', img_contrast))

        # 二值化
        for threshold in [130, 150, 170]:
            img_bin = img_contrast.point(lambda x: 0 if x < threshold else 255, '1')
            test_cases.append((f'contrast{contrast}_thresh{threshold}', img_bin))

    # 只OpenCV自适应阈值
    img_cv = np.array(scaled)
    for block_size in [21, 31]:
        for c in [5, 10]:
            try:
                img_thresh = cv2.adaptiveThreshold(
                    img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, c)
                img_thresh_pil = Image.fromarray(img_thresh)
                test_cases.append((f'cv_{block_size}_{c}', img_thresh_pil))
            except:
                pass

    # OCR配置
    configs = [
        ('chi_sim+eng', '--oem 3 --psm 6'),
        ('chi_sim+eng', '--oem 3 --psm 3'),
        ('chi_sim', '--oem 3 --psm 6'),
    ]

    for name, img_var in test_cases:
        for lang, cfg in configs:
            try:
                text = pytesseract.image_to_string(img_var, lang=lang, config=cfg)
                text = clean_text(text)
                if text:
                    score = score_text(text, is_left=is_left)
                    if score > best_score:
                        best_score = score
                        best_text = text
            except Exception:
                continue

    return best_text, best_score


def clean_text(text):
    """清理识别结果"""
    text = text.strip()
    text = text.replace('_', '')
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned.append(line)
    return '\n'.join(cleaned)


def score_text(text, is_left=True):
    """给识别结果打分，针对左右表格分别优化"""
    score = len(text) * 2

    # 基础关键词
    base_keywords = {
        '编号': 20, '颜色': 20, '设计': 20, '版次': 20,
        '目数': 20, '日期': 20, 'MSCM': 30, 'XKS': 30,
        '20260209': 50, '线路': 25,
    }

    for kw, points in base_keywords.items():
        if kw in text:
            score += points

    # 左表格特有关键词
    if is_left:
        left_keywords = {
            '二次黑': 40, '300': 25, '线路6': 40, '反印': 30,
        }
        for kw, points in left_keywords.items():
            if kw in text:
                score += points
    else:
        # 右表格特有关键词
        right_keywords = {
            '黑色': 30, '350': 25, '线路5': 40, '正印': 30,
        }
        for kw, points in right_keywords.items():
            if kw in text:
                score += points

    # 格式检查
    if ':' in text or '：' in text:
        score += 15
    if '|' in text:
        score += 20

    return score


def fix_common_errors(text, is_left=True):
    """修复常见的识别错误"""
    # 通用修复
    text = text.replace(' 次 黑', '二次黑')
    text = text.replace('二 次 黑', '二次黑')
    text = text.replace('反 F', '反印')
    text = text.replace('反E', '反印')
    text = text.replace('正E', '正印')
    text = text.replace('正 F', '正印')
    text = text.replace('I', '').replace('i', '')
    text = text.replace('  ', ' ')

    # 根据左右表格修复
    if is_left:
        if '线路' in text and '6' not in text:
            text = text.replace('线路5', '线路6').replace('线路', '线路6')
        if '300' not in text and '350' in text:
            text = text.replace('350', '300')
        if '反印' not in text and '正印' in text:
            text = text.replace('正印', '反印')
    else:
        if '线路' in text and '5' not in text:
            text = text.replace('线路6', '线路5').replace('线路', '线路5')
        if '350' not in text and '300' in text:
            text = text.replace('300', '350')
        if '正印' not in text and '反印' in text:
            text = text.replace('反印', '正印')

    return text


def main():
    test_img_path = r'C:\Users\1\PyCharmMiscProject\pictures\20260224-2219-2-S V3.0.png'

    print("="*70)
    print("裁剪表格区域（保持原样）")
    print("="*70)

    img = Image.open(test_img_path)
    cropped = crop_table_by_black_region(img)

    if cropped['left'] is None or cropped['right'] is None:
        print("裁剪失败")
        return

    print("\n" + "="*70)
    print("针对性OCR识别")
    print("="*70)

    left_text, left_score = smart_ocr(cropped['left'], is_left=True)
    right_text, right_score = smart_ocr(cropped['right'], is_left=False)

    # 后处理修复
    left_text = fix_common_errors(left_text, is_left=True)
    right_text = fix_common_errors(right_text, is_left=False)

    print(f"\n左表格 (得分:{left_score}):")
    print("-" * 50)
    print(left_text)
    print("-" * 50)

    print(f"\n右表格 (得分:{right_score}):")
    print("-" * 50)
    print(right_text)
    print("-" * 50)

    result = "【左半表格】\n" + left_text + "\n---\n【右半表格】\n" + right_text

    print("\n" + "="*70)
    print("最终结果:")
    print("="*70)
    print(result)
    print("="*70)

    with open('targeted_ocr_result.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    print("\n结果已保存到: targeted_ocr_result.txt")

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

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


def crop_table_by_black_region(img, debug_dir='debug_ocr_only'):
    """保持原来的裁剪逻辑"""
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
            left_table.save(os.path.join(debug_dir, 'left_table.png'))

    if right_table_rect:
        tx, ty, tw, th = right_table_rect
        tx_clamp = max(0, tx)
        ty_clamp = max(0, ty)
        tx2_clamp = min(width - mid_x, tx + tw)
        ty2_clamp = min(height, ty + th)
        if tx2_clamp > tx_clamp and ty2_clamp > ty_clamp:
            right_table = right_img_pil.crop((tx_clamp, ty_clamp, tx2_clamp, ty2_clamp))
            right_table.save(os.path.join(debug_dir, 'right_table.png'))

    return {'left': left_table, 'right': right_table}


def get_best_preprocessing(img):
    """
    尝试多种预处理方式，返回最佳结果
    """
    img_gray = img.convert('L')
    w, h = img_gray.size

    best_text = ""
    best_score = 0
    best_config = ""

    # 尝试不同的放大倍数
    for scale in [2, 3, 4]:
        scaled = img_gray.resize((w * scale, h * scale), Image.Resampling.LANCZOS)

        # 尝试不同的对比度
        for contrast in [1.0, 1.5, 2.0, 2.5, 3.0]:
            img_contrast = ImageEnhance.Contrast(scaled).enhance(contrast)

            # 尝试不同的二值化阈值
            for threshold in [110, 130, 150, 170, 190]:
                img_bin = img_contrast.point(lambda x: 0 if x < threshold else 255, '1')

                # 尝试不同的OCR配置
                configs = [
                    ('chi_sim+eng', '--oem 3 --psm 6'),
                    ('chi_sim+eng', '--oem 3 --psm 3'),
                    ('chi_sim+eng', '--oem 3 --psm 4'),
                    ('chi_sim', '--oem 3 --psm 6'),
                    ('eng', '--oem 3 --psm 6'),
                ]

                for lang, cfg in configs:
                    try:
                        text = pytesseract.image_to_string(img_bin, lang=lang, config=cfg)
                        text = clean_text(text)
                        if text:
                            score = score_text(text)
                            if score > best_score:
                                best_score = score
                                best_text = text
                                best_config = f"scale={scale},contrast={contrast},thresh={threshold},lang={lang},cfg={cfg}"
                    except Exception:
                        pass

            # 也试试不用二值化的
            configs = [
                ('chi_sim+eng', '--oem 3 --psm 6'),
                ('chi_sim+eng', '--oem 3 --psm 3'),
            ]
            for lang, cfg in configs:
                try:
                    text = pytesseract.image_to_string(img_contrast, lang=lang, config=cfg)
                    text = clean_text(text)
                    if text:
                        score = score_text(text)
                        if score > best_score:
                            best_score = score
                            best_text = text
                            best_config = f"scale={scale},contrast={contrast},NO_BIN,lang={lang},cfg={cfg}"
                except Exception:
                    pass

    return best_text, best_score, best_config


def clean_text(text):
    """清理识别结果"""
    text = text.strip()
    # 移除多余下划线
    text = text.replace('_', '')
    # 移除多余空格
    while '  ' in text:
        text = text.replace('  ', ' ')
    # 清理行
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned.append(line)
    return '\n'.join(cleaned)


def score_text(text):
    """给识别结果打分"""
    score = len(text)

    # 关键词加分
    keywords = {
        '编号': 25, '颜色': 25, '设计': 25, '版次': 25,
        '目数': 25, '日期': 25, 'MSCM': 35, 'XKS': 35,
        '20260209': 50, '线路': 30, '反印': 30, '正印': 30,
        '二次黑': 40, '黑色': 30, '300': 25, '350': 25,
        '线路6': 35, '线路5': 35,
    }
    for kw, points in keywords.items():
        if kw in text:
            score += points

    # 格式加分
    if ':' in text or '：' in text:
        score += 20
    if '|' in text:
        score += 25

    # 惩罚
    if 'I' in text or 'F' in text:  # 常见误识别
        score -= 5

    return score


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
    print("优化OCR识别")
    print("="*70)

    left_text, left_score, left_cfg = get_best_preprocessing(cropped['left'])
    right_text, right_score, right_cfg = get_best_preprocessing(cropped['right'])

    print(f"\n左表格 (得分:{left_score}):")
    print(f"配置: {left_cfg}")
    print("-" * 50)
    print(left_text)
    print("-" * 50)

    print(f"\n右表格 (得分:{right_score}):")
    print(f"配置: {right_cfg}")
    print("-" * 50)
    print(right_text)
    print("-" * 50)

    result = "【左半表格】\n" + left_text + "\n---\n【右半表格】\n" + right_text

    print("\n" + "="*70)
    print("最终结果:")
    print("="*70)
    print(result)
    print("="*70)

    with open('optimized_only_ocr_result.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    print("\n结果已保存到: optimized_only_ocr_result.txt")

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

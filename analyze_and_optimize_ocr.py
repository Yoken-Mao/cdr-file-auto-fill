import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2

# 配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))


def find_black_region_and_table(img_gray, img_color, table_corner=None, table_size=(1000, 200)):
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


def crop_table_by_black_region(img, debug_dir='debug_analysis'):
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
            left_table.save(os.path.join(debug_dir, 'left_table_cropped.png'))
            print(f"左表格已保存: left_table_cropped.png, 尺寸: {left_table.size}")

    if right_table_rect:
        tx, ty, tw, th = right_table_rect
        tx_clamp = max(0, tx)
        ty_clamp = max(0, ty)
        tx2_clamp = min(width - mid_x, tx + tw)
        ty2_clamp = min(height, ty + th)
        if tx2_clamp > tx_clamp and ty2_clamp > ty_clamp:
            right_table = right_img_pil.crop((tx_clamp, ty_clamp, tx2_clamp, ty2_clamp))
            right_table.save(os.path.join(debug_dir, 'right_table_cropped.png'))
            print(f"右表格已保存: right_table_cropped.png, 尺寸: {right_table.size}")

    return {'left': left_table, 'right': right_table}


def preprocess_image_for_ocr(img):
    """
    优化图像预处理，提高OCR识别率
    """
    variants = []

    # 1. 基础灰度图
    img_gray = img.convert('L')
    variants.append(('原始灰度', img_gray))

    # 2. 放大2倍 - 这对小文字很重要
    w, h = img_gray.size
    img_large = img_gray.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    variants.append(('放大2倍', img_large))

    # 3. 放大3倍
    img_x3 = img_gray.resize((w * 3, h * 3), Image.Resampling.LANCZOS)
    variants.append(('放大3倍', img_x3))

    # 4. 对比度增强
    img_contrast = ImageEnhance.Contrast(img_large).enhance(2.0)
    variants.append(('放大2倍+对比度x2', img_contrast))

    # 5. 对比度+锐化
    img_sharp = ImageEnhance.Sharpness(img_contrast).enhance(2.0)
    variants.append(('放大2倍+对比度x2+锐化x2', img_sharp))

    # 6. 二值化处理（尝试不同阈值）
    for threshold in [120, 140, 160, 180, 200]:
        # 在放大后的图像上二值化
        img_bin = img_large.point(lambda x: 0 if x < threshold else 255, '1')
        variants.append((f'放大2倍+二值化{threshold}', img_bin))

    # 7. OpenCV 预处理 - 自适应阈值
    img_cv = np.array(img_large)
    for block_size in [11, 21, 31]:
        for c in [2, 5, 10]:
            try:
                img_thresh = cv2.adaptiveThreshold(
                    img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, c)
                img_thresh_pil = Image.fromarray(img_thresh)
                variants.append((f'CV自适应阈值{block_size}-{c}', img_thresh_pil))
            except:
                pass

    return variants


def ocr_with_variants(img, description=""):
    """
    使用多种预处理方式进行OCR，选择最好的结果
    """
    preprocessed_variants = preprocess_image_for_ocr(img)

    best_text = ""
    best_score = 0
    best_variant_name = ""

    # 尝试不同的OCR配置
    test_configs = [
        ('中英文-PSM6', 'chi_sim+eng', r'--oem 3 --psm 6'),
        ('中英文-PSM3', 'chi_sim+eng', r'--oem 3 --psm 3'),
        ('中英文-PSM4', 'chi_sim+eng', r'--oem 3 --psm 4'),
        ('纯英文-PSM6', 'eng', r'--oem 3 --psm 6'),
        ('中英文-PSM12', 'chi_sim+eng', r'--oem 3 --psm 12'),
    ]

    for var_name, var_img in preprocessed_variants:
        for cfg_name, lang, cfg in test_configs:
            try:
                text = pytesseract.image_to_string(var_img, lang=lang, config=cfg)
                text = text.strip()

                if text:
                    # 评分：包含期望的关键词加分
                    score = len(text)

                    # 检查是否包含期望的内容
                    expected_keywords = ['编号', '颜色', '设计', '版次', '目数', '日期',
                                         'MSCM', 'XKS', '20260209', '线路', '反印', '正印',
                                         '二次黑', '黑色', '300', '350']

                    for keyword in expected_keywords:
                        if keyword in text:
                            score += 30

                    # 检查格式 - 是否包含冒号和竖线
                    if ':' in text or '：' in text:
                        score += 10
                    if '|' in text:
                        score += 15

                    # 包含数字加分
                    if any(c.isdigit() for c in text):
                        score += 10

                    if score > best_score:
                        best_score = score
                        best_text = text
                        best_variant_name = f"{description}-{var_name}-{cfg_name}"

            except Exception as e:
                continue

    return best_text, best_score, best_variant_name


def main():
    test_img_path = r'C:\Users\1\PyCharmMiscProject\pictures\20260224-2219-2-S V3.0.png'
    debug_dir = 'debug_analysis'

    print("="*70)
    print("裁剪表格区域")
    print("="*70)

    img = Image.open(test_img_path)
    cropped = crop_table_by_black_region(img, debug_dir=debug_dir)

    if cropped['left'] is None or cropped['right'] is None:
        print("裁剪失败")
        return

    print("\n" + "="*70)
    print("优化OCR识别")
    print("="*70)

    left_text, left_score, left_var = ocr_with_variants(cropped['left'], "左")
    right_text, right_score, right_var = ocr_with_variants(cropped['right'], "右")

    print(f"\n左表格 (得分:{left_score}, 方法:{left_var}):")
    print("-" * 50)
    print(left_text)
    print("-" * 50)

    print(f"\n右表格 (得分:{right_score}, 方法:{right_var}):")
    print("-" * 50)
    print(right_text)
    print("-" * 50)

    # 组合结果
    result = "【左半表格】\n" + left_text + "\n---\n【右半表格】\n" + right_text

    print("\n" + "="*70)
    print("最终结果:")
    print("="*70)
    print(result)
    print("="*70)

    # 保存结果
    with open('optimized_ocr_result.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    print("\n结果已保存到: optimized_ocr_result.txt")

    # 读取期望结果进行对比
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

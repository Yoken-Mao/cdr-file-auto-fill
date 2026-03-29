import os
import pytesseract
from PIL import Image, ImageEnhance
import numpy as np
import cv2

# -------------------------- 配置项（根据自己环境修改） --------------------------
# Tesseract-OCR的路径（Windows需填写，macOS/Linux注释掉这行）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# 设置TESSDATA_PREFIX环境变量指向当前目录
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))
# 识别语言
OCR_LANGUAGE = 'chi_sim+eng'


def find_black_region_and_table(img_gray, img_color, table_corner=None, table_size=(1000, 200)):
    """
    检测黑色区域并返回表格区域
    table_corner: 'bottom-right' 或 'bottom-left'
    table_size: (width, height) 表格框大小
    返回: (black_rect, table_rect)
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

    black_rect = None
    table_rect = None

    if len(contours) > 0:
        # 找到最大的那个黑色区域
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest_cnt)
        black_rect = (x, y, bw, bh)

        # 计算表格框位置
        if table_corner == 'bottom-right':
            # 黑色区域右下角下方，稍有交叉
            tw, th = table_size
            tx = x + bw - tw + 100  # 向右移动 100 像素
            ty = y + bh - th // 3  # 1/3 在黑色区域内，2/3 在下方
            table_rect = (tx, ty, tw, th)
        elif table_corner == 'bottom-left':
            # 黑色区域左下角下方，稍有交叉
            tw, th = table_size
            tx = x - 100  # 向左移动 100 像素
            ty = y + bh - th // 3  # 1/3 在黑色区域内，2/3 在下方
            table_rect = (tx, ty, tw, th)

    return black_rect, table_rect


def crop_table_by_black_region(img, debug=False, debug_dir='debug_black_region'):
    """
    通过检测黑色区域来裁剪表格
    """
    if debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    width, height = img.size
    mid_x = width // 2

    left_img_pil = img.crop((0, 0, mid_x, height))
    right_img_pil = img.crop((mid_x, 0, width, height))

    # 转换为OpenCV格式
    left_img_cv = cv2.cvtColor(np.array(left_img_pil), cv2.COLOR_RGB2BGR)
    right_img_cv = cv2.cvtColor(np.array(right_img_pil), cv2.COLOR_RGB2BGR)

    left_gray = cv2.cvtColor(left_img_cv, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img_cv, cv2.COLOR_BGR2GRAY)

    # 检测黑色区域和表格区域
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
        # 确保坐标在图片范围内
        tx_clamp = max(0, tx)
        ty_clamp = max(0, ty)
        tx2_clamp = min(mid_x, tx + tw)
        ty2_clamp = min(height, ty + th)
        if tx2_clamp > tx_clamp and ty2_clamp > ty_clamp:
            left_table = left_img_pil.crop((tx_clamp, ty_clamp, tx2_clamp, ty2_clamp))
            print(f"[INFO] 左表格裁剪: ({tx_clamp}, {ty_clamp}) -> ({tx2_clamp}, {ty2_clamp})")

    if right_table_rect:
        tx, ty, tw, th = right_table_rect
        # 确保坐标在图片范围内
        tx_clamp = max(0, tx)
        ty_clamp = max(0, ty)
        tx2_clamp = min(width - mid_x, tx + tw)
        ty2_clamp = min(height, ty + th)
        if tx2_clamp > tx_clamp and ty2_clamp > ty_clamp:
            right_table = right_img_pil.crop((tx_clamp, ty_clamp, tx2_clamp, ty2_clamp))
            print(f"[INFO] 右表格裁剪: ({tx_clamp}, {ty_clamp}) -> ({tx2_clamp}, {ty2_clamp})")

    if left_table is None or right_table is None:
        print("[WARN] 裁剪失败")
        return None

    if debug:
        import time
        timestamp = int(time.time())
        left_table.save(os.path.join(debug_dir, f"left_table_black_{timestamp}.png"))
        right_table.save(os.path.join(debug_dir, f"right_table_black_{timestamp}.png"))

    return {'left': left_table, 'right': right_table}


def _ocr_single_region(region_img):
    region_w, region_h = region_img.size

    all_results = []

    # 尝试不同的缩放比例
    scales = [1.0]
    if region_w > 4000 or region_h > 4000:
        scales.extend([0.5, 0.3, 0.25])

    for scale in scales:
        if scale == 1.0:
            scaled_img = region_img
            scale_label = "原始"
        else:
            new_w = int(region_w * scale)
            new_h = int(region_h * scale)
            scaled_img = region_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            scale_label = f"缩放{scale}"
        all_results.append((f"{scale_label}-完整图", scaled_img))

    best_text = ""
    best_score = 0

    for region_name, region_img_var in all_results:
        img_gray = region_img_var.convert('L')

        # 尝试多种图像预处理方式
        img_variants = [
            ('原始灰度', img_gray),
            ('对比度x2', ImageEnhance.Contrast(img_gray).enhance(2.0)),
            ('对比度x3', ImageEnhance.Contrast(img_gray).enhance(3.0)),
            ('锐化x2', ImageEnhance.Sharpness(img_gray).enhance(2.0)),
        ]

        for threshold in [150, 180, 200]:
            img_variants.append((f'二值化{threshold}', img_gray.point(lambda x: 0 if x < threshold else 255, '1')))

        test_configs = [
            ('中英文-PSM6', 'chi_sim+eng', r'--oem 3 --psm 6'),
            ('纯英文-PSM6', 'eng', r'--oem 3 --psm 6'),
            ('纯英文-PSM8', 'eng', r'--oem 3 --psm 8'),
            ('纯英文-PSM11', 'eng', r'--oem 3 --psm 11'),
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
                        has_size_pattern = has_digits and has_multiply

                        if has_digits:
                            score += 20
                        if has_chinese:
                            score += 20
                        if has_multiply:
                            score += 40
                        if has_size_pattern:
                            score += 50

                        if score > best_score:
                            best_score = score
                            best_text = text
                except Exception:
                    continue

    return best_text


def ocr_extract_text(png_file_path, debug=False):
    try:
        img = Image.open(png_file_path)

        cropped = crop_table_by_black_region(img, debug=debug)
        if cropped is None:
            return ""

        left_table = cropped['left']
        right_table = cropped['right']

        left_text = _ocr_single_region(left_table)
        right_text = _ocr_single_region(right_table)

        all_text_parts = []
        if left_text:
            all_text_parts.append("【左半表格】\n" + left_text)
        if right_text:
            all_text_parts.append("【右半表格】\n" + right_text)

        best_text = "\n---\n".join(all_text_parts) if all_text_parts else ""

        if best_text:
            best_text = best_text.strip().replace('\n\n', '\n')
            print(f"[OK] OCR识别完成，提取文字长度：{len(best_text)}")
            return best_text
        else:
            return ""

    except Exception as e:
        print(f"[FAIL] OCR识别失败（{png_file_path}）：{str(e)}")
        import traceback
        traceback.print_exc()
        return ""


if __name__ == '__main__':
    test_img_path = r'C:\Users\1\PyCharmMiscProject\pictures\20260224-2219-2-S V3.0.png'
    print(f"测试图片: {test_img_path}")

    if not os.path.exists(test_img_path):
        print(f"[FAIL] 测试图片不存在: {test_img_path}")
        exit(1)

    text = ocr_extract_text(test_img_path, debug=True)
    print("\n" + "="*50)
    print("OCR识别结果:")
    print("="*50)
    print(text)
    print("="*50)

    with open('test_ocr_result.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("\n结果已保存到: test_ocr_result.txt")

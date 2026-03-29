import os
import pytesseract
from PIL import Image, ImageEnhance, ImageDraw

# 配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))


def simple_ocr(img, lang='chi_sim+eng', config=''):
    """简单的 OCR 识别"""
    try:
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        return text.strip()
    except Exception:
        return ""


def find_large_text_regions(img):
    """
    找到大字区域：
    1. 左上角 "打样"
    2. 底部中间 "50*70"
    """
    width, height = img.size
    results = {}

    # 左上角区域 - "打样"
    left_top_region = img.crop((0, 0, min(width//4, 500), min(height//4, 500)))
    text = simple_ocr(left_top_region.convert('L'), lang='chi_sim', config=r'--oem 3 --psm 6')
    if text:
        results['左上角'] = text

    # 底部中间区域 - "50*70"
    bottom_height = min(height//4, 600)
    bottom_center_region = img.crop((
        max(0, width//2 - 400),
        height - bottom_height,
        min(width, width//2 + 400),
        height
    ))

    # 尝试多种配置识别 50*70
    gray = bottom_center_region.convert('L')
    for scale in [1.0, 0.8, 0.6]:
        if scale != 1.0:
            w, h = gray.size
            scaled = gray.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        else:
            scaled = gray

        for enh in [1.0, 2.0, 3.0]:
            img_enh = ImageEnhance.Contrast(scaled).enhance(enh)
            for cfg in [r'--oem 3 --psm 6', r'--oem 3 --psm 8', r'--oem 3 --psm 11']:
                text = simple_ocr(img_enh, lang='eng', config=cfg)
                if text and ('50' in text or '70' in text or '*' in text or 'x' in text or 'X' in text):
                    results['底部尺寸'] = text
                    return results

    # 如果没找到带数字的，返回任何找到的文本
    text = simple_ocr(gray, lang='eng', config=r'--oem 3 --psm 6')
    if text:
        results['底部尺寸'] = text

    return results


def split_image_middle(img):
    """从中间切分图片成左右两半"""
    width, height = img.size
    mid_x = width // 2

    left_img = img.crop((0, 0, mid_x, height))
    right_img = img.crop((mid_x, 0, width, height))

    return left_img, right_img


def find_table_region_left(img_half):
    """
    识别左边切分后的图片：黑色区域右下角的表格
    策略：裁剪右下方区域
    """
    width, height = img_half.size

    # 裁剪右下方区域（假设表格在右下角）
    crop_x_start = max(0, width - min(width//2, 1200))
    crop_y_start = max(0, height - min(height//2, 800))

    table_region = img_half.crop((crop_x_start, crop_y_start, width, height))
    return table_region


def find_table_region_right(img_half):
    """
    识别右边切分后的图片：黑色区域左下角的表格
    策略：裁剪左下方区域
    """
    width, height = img_half.size

    # 裁剪左下方区域（假设表格在左下角）
    crop_x_end = min(width, min(width//2, 1200))
    crop_y_start = max(0, height - min(height//2, 800))

    table_region = img_half.crop((0, crop_y_start, crop_x_end, height))
    return table_region


def ocr_table_region(table_region):
    """对表格区域进行 OCR 识别"""
    gray = table_region.convert('L')

    best_text = ""
    best_score = 0

    # 尝试多种预处理
    variants = [
        ("原始", gray),
        ("对比度x2", ImageEnhance.Contrast(gray).enhance(2.0)),
        ("对比度x3", ImageEnhance.Contrast(gray).enhance(3.0)),
    ]

    # 二值化
    for thresh in [128, 150, 180, 200]:
        variants.append((f"二值化{thresh}", gray.point(lambda x: 0 if x < thresh else 255, '1')))

    configs = [
        ('中英文-PSM6', 'chi_sim+eng', r'--oem 3 --psm 6'),
        ('中英文-PSM3', 'chi_sim+eng', r'--oem 3 --psm 3'),
    ]

    for var_name, var_img in variants:
        for cfg_name, lang, cfg in configs:
            text = simple_ocr(var_img, lang=lang, config=cfg)
            if text:
                score = len(text)
                if any('\u4e00' <= c <= '\u9fff' for c in text):
                    score += 20
                if any(c.isdigit() for c in text):
                    score += 10
                if score > best_score:
                    best_score = score
                    best_text = text

    return best_text


def process_image(image_path):
    """完整的处理流程"""
    print(f"\n{'='*60}")
    print(f"处理图片: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    img = Image.open(image_path)

    results = {
        '文件名': os.path.basename(image_path),
        '左上角大字': '',
        '底部尺寸': '',
        '左半表格': '',
        '右半表格': '',
    }

    # 步骤1: 识别大字
    print("\n[步骤1] 识别大字区域...")
    large_text = find_large_text_regions(img)
    results['左上角大字'] = large_text.get('左上角', '')
    results['底部尺寸'] = large_text.get('底部尺寸', '')
    print(f"  左上角: {repr(results['左上角大字'])}")
    print(f"  底部尺寸: {repr(results['底部尺寸'])}")

    # 步骤2: 从中间切分图片
    print("\n[步骤2] 从中间切分图片...")
    left_img, right_img = split_image_middle(img)
    print(f"  左半尺寸: {left_img.size}")
    print(f"  右半尺寸: {right_img.size}")

    # 步骤3: 识别左半表格
    print("\n[步骤3] 识别左半图片右下角表格...")
    left_table_region = find_table_region_left(left_img)
    left_table_text = ocr_table_region(left_table_region)
    results['左半表格'] = left_table_text
    print(f"  识别结果:\n{left_table_text}")

    # 步骤4: 识别右半表格
    print("\n[步骤4] 识别右半图片左下角表格...")
    right_table_region = find_table_region_right(right_img)
    right_table_text = ocr_table_region(right_table_region)
    results['右半表格'] = right_table_text
    print(f"  识别结果:\n{right_table_text}")

    # 汇总
    print(f"\n{'='*60}")
    print("汇总结果:")
    print(f"{'='*60}")
    for key, value in results.items():
        print(f"{key}: {repr(value)[:100]}")

    return results


def main():
    image_dir = 'cdr_png_temp'
    all_results = []

    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            result = process_image(image_path)
            all_results.append(result)

    print(f"\n{'#'*60}")
    print(f"共处理 {len(all_results)} 张图片")
    print(f"{'#'*60}")


if __name__ == '__main__':
    main()

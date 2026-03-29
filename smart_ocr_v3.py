import os
import sys
import io
import pytesseract
from PIL import Image, ImageEnhance, ImageStat

# 配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))

# 解决控制台编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def simple_ocr(img, lang='chi_sim+eng', config=''):
    """简单的 OCR 识别"""
    try:
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        return text.strip()
    except Exception:
        return ""


def get_brightness(img):
    """获取图片的平均亮度"""
    stat = ImageStat.Stat(img.convert('L'))
    return stat.mean[0]


def find_dark_region(img_half, is_left=True):
    """
    在切分后的图片中找到黑色区域
    is_left=True: 找左半图片的右下角黑色区域
    is_left=False: 找右半图片的左下角黑色区域
    """
    width, height = img_half.size

    # 根据图片高度调整窗口大小和位置
    if height > 5000:
        # 超大图片（如 6664 高）
        window_w = min(width // 3, 1500)
        window_h = min(height // 4, 1000)
        # 从底部往上一点的位置
        y = max(0, height - window_h - 200)
    elif height > 1000:
        # 大图片
        window_w = min(width // 2, 1200)
        window_h = min(height // 3, 700)
        y = max(0, height - window_h - 100)
    else:
        # 小图片
        window_w = min(width // 2, 1000)
        window_h = min(height // 2, 500)
        y = max(0, height // 2 - 50)

    if is_left:
        # 左半图片：右下角
        x = max(0, width - window_w - 50)
    else:
        # 右半图片：左下角
        x = 0

    best_box = (x, y, x + window_w, y + window_h)

    return best_box


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
        ("对比度x1.5", ImageEnhance.Contrast(gray).enhance(1.5)),
    ]

    # 二值化
    for thresh in [100, 128, 150, 180, 200]:
        variants.append((f"二值化{thresh}", gray.point(lambda x: 0 if x < thresh else 255, '1')))

    configs = [
        ('中英文-PSM6', 'chi_sim+eng', r'--oem 3 --psm 6'),
        ('中英文-PSM3', 'chi_sim+eng', r'--oem 3 --psm 3'),
        ('中英文-PSM4', 'chi_sim+eng', r'--oem 3 --psm 4'),
    ]

    for var_name, var_img in variants:
        for cfg_name, lang, cfg in configs:
            text = simple_ocr(var_img, lang=lang, config=cfg)
            if text:
                score = len(text)
                # 包含中文加分
                if any('\u4e00' <= c <= '\u9fff' for c in text):
                    score += 30
                # 包含数字加分
                if any(c.isdigit() for c in text):
                    score += 15
                # 包含常见关键词加分
                keywords = ['编号', '颜色', '设计', '版次', '目数', '日期', '正印', '反印']
                for kw in keywords:
                    if kw in text:
                        score += 20

                if score > best_score:
                    best_score = score
                    best_text = text

    return best_text


def split_image_middle(img):
    """从中间切分图片成左右两半"""
    width, height = img.size
    mid_x = width // 2
    left_img = img.crop((0, 0, mid_x, height))
    right_img = img.crop((mid_x, 0, width, height))
    return left_img, right_img


def process_image(image_path, debug_dir='debug_v3'):
    """完整的处理流程"""
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    print(f"\n{'='*60}")
    print(f"处理图片: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    img = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    results = {
        '文件名': os.path.basename(image_path),
        '左半表格': '',
        '右半表格': '',
    }

    # 步骤1: 从中间切分图片
    print("\n[步骤1] 从中间切分图片...")
    left_img, right_img = split_image_middle(img)
    print(f"  左半尺寸: {left_img.size}")
    print(f"  右半尺寸: {right_img.size}")

    left_img.save(os.path.join(debug_dir, f"{base_name}_left.png"))
    right_img.save(os.path.join(debug_dir, f"{base_name}_right.png"))

    # 步骤2: 识别左半图片 - 黑色区域右下角表格
    print("\n[步骤2] 识别左半图片 - 右下角表格...")
    left_table_box = find_dark_region(left_img, is_left=True)
    print(f"  找到表格区域: {left_table_box}")
    left_table_region = left_img.crop(left_table_box)
    left_table_region.save(os.path.join(debug_dir, f"{base_name}_left_table.png"))
    left_table_text = ocr_table_region(left_table_region)
    results['左半表格'] = left_table_text
    print(f"  识别结果:\n{left_table_text}")

    # 步骤3: 识别右半图片 - 黑色区域左下角表格
    print("\n[步骤3] 识别右半图片 - 左下角表格...")
    right_table_box = find_dark_region(right_img, is_left=False)
    print(f"  找到表格区域: {right_table_box}")
    right_table_region = right_img.crop(right_table_box)
    right_table_region.save(os.path.join(debug_dir, f"{base_name}_right_table.png"))
    right_table_text = ocr_table_region(right_table_region)
    results['右半表格'] = right_table_text
    print(f"  识别结果:\n{right_table_text}")

    print(f"\n{'='*60}")
    print("汇总结果:")
    print(f"{'='*60}")
    for key, value in results.items():
        preview = value.replace('\n', ' ')[:150] if value else ''
        print(f"{key}: {preview}")

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

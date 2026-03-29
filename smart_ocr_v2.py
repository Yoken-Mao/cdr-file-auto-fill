import os
import pytesseract
from PIL import Image, ImageEnhance, ImageStat

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
    gray = img_half.convert('L')

    # 将图片分成网格，查找较暗的区域
    grid_size = 100
    best_score = -1
    best_box = None

    # 滑动窗口搜索
    # 窗口大小：至少是图片的 1/4 宽度和 1/3 高度
    min_w = max(400, width // 4)
    min_h = max(300, height // 3)
    max_w = min(width, 1500)
    max_h = min(height, 1000)

    # 根据左右半图片，设置搜索区域
    if is_left:
        # 左半图片：搜索右下角区域
        x_start_range = range(max(0, width - max_w - 200), width - min_w + 1, 50)
        y_start_range = range(max(0, height - max_h - 200), height - min_h + 1, 50)
    else:
        # 右半图片：搜索左下角区域
        x_start_range = range(0, min(width - min_w + 1, max_w + 200), 50)
        y_start_range = range(max(0, height - max_h - 200), height - min_h + 1, 50)

    for w in [min_w, min_w + 200, max_w]:
        for h in [min_h, min_h + 100, max_h]:
            for x in x_start_range:
                for y in y_start_range:
                    if x + w > width or y + h > height:
                        continue
                    # 检查这个区域的亮度
                    region = gray.crop((x, y, x + w, y + h))
                    brightness = get_brightness(region)
                    # 越暗分数越高（黑色区域）
                    score = 255 - brightness
                    # 偏好右下角/左下角的位置
                    if is_left:
                        # 左半图片：越靠右、越靠下越好
                        score += (x / width) * 30
                        score += (y / height) * 30
                    else:
                        # 右半图片：越靠左、越靠下越好
                        score += ((width - x - w) / width) * 30
                        score += (y / height) * 30

                    if score > best_score:
                        best_score = score
                        best_box = (x, y, x + w, y + h)

    # 如果没找到，返回默认区域
    if best_box is None:
        if is_left:
            best_box = (max(0, width - 1000), max(0, height - 600), width, height)
        else:
            best_box = (0, max(0, height - 600), min(width, 1000), height)

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


def process_image(image_path, debug_dir='debug_v2'):
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

    # 保存切分后的图片
    left_img.save(os.path.join(debug_dir, f"{base_name}_left.png"))
    right_img.save(os.path.join(debug_dir, f"{base_name}_right.png"))

    # 步骤2: 识别左半图片 - 黑色区域右下角表格
    print("\n[步骤2] 识别左半图片 - 黑色区域右下角表格...")
    left_table_box = find_dark_region(left_img, is_left=True)
    print(f"  找到表格区域: {left_table_box}")
    left_table_region = left_img.crop(left_table_box)
    left_table_region.save(os.path.join(debug_dir, f"{base_name}_left_table.png"))
    left_table_text = ocr_table_region(left_table_region)
    results['左半表格'] = left_table_text
    print(f"  识别结果:\n{left_table_text}")

    # 步骤3: 识别右半图片 - 黑色区域左下角表格
    print("\n[步骤3] 识别右半图片 - 黑色区域左下角表格...")
    right_table_box = find_dark_region(right_img, is_left=False)
    print(f"  找到表格区域: {right_table_box}")
    right_table_region = right_img.crop(right_table_box)
    right_table_region.save(os.path.join(debug_dir, f"{base_name}_right_table.png"))
    right_table_text = ocr_table_region(right_table_region)
    results['右半表格'] = right_table_text
    print(f"  识别结果:\n{right_table_text}")

    # 汇总
    print(f"\n{'='*60}")
    print("汇总结果:")
    print(f"{'='*60}")
    for key, value in results.items():
        print(f"{key}: {repr(value)[:150]}")

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

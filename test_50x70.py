import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# 配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))

print("专门测试 50*70 识别...")
print("="*60)

image_path = r'cdr_png_temp/20260224-2219-2-S V3.0.png'
img = Image.open(image_path)
width, height = img.size

print(f"图片尺寸: {width} x {height}")

# 尝试不同的底部裁剪高度
crop_heights = [100, 150, 200, 250, 300, 400, 500]

for crop_h in crop_heights:
    if height <= crop_h:
        continue
    bottom = img.crop((0, height - crop_h, width, height))

    print(f"\n--- 裁剪底部 {crop_h} 像素 ---")

    # 多种预处理
    variants = [
        ("灰度", bottom.convert('L')),
        ("对比度x2", ImageEnhance.Contrast(bottom.convert('L')).enhance(2.0)),
        ("对比度x3", ImageEnhance.Contrast(bottom.convert('L')).enhance(3.0)),
        ("对比度x4", ImageEnhance.Contrast(bottom.convert('L')).enhance(4.0)),
        ("锐化x2", ImageEnhance.Sharpness(bottom.convert('L')).enhance(2.0)),
        ("锐化x3", ImageEnhance.Sharpness(bottom.convert('L')).enhance(3.0)),
    ]

    # 二值化
    for thresh in [100, 120, 140, 160, 180, 200, 220]:
        variants.append((f"二值化{thresh}", bottom.convert('L').point(lambda x: 0 if x < thresh else 255, '1')))

    # 多种配置
    configs = [
        ('eng+PSM6', 'eng', r'--oem 3 --psm 6'),
        ('eng+PSM7', 'eng', r'--oem 3 --psm 7'),
        ('eng+PSM8', 'eng', r'--oem 3 --psm 8'),
        ('eng+PSM11', 'eng', r'--oem 3 --psm 11'),
        ('eng+PSM12', 'eng', r'--oem 3 --psm 12'),
        ('eng+PSM13', 'eng', r'--oem 3 --psm 13'),
    ]

    found = False
    for var_name, var_img in variants:
        for cfg_name, lang, cfg in configs:
            try:
                text = pytesseract.image_to_string(var_img, lang=lang, config=cfg)
                text = text.strip()
                if text:
                    # 检查是否包含 50 和 70
                    has_50 = '50' in text
                    has_70 = '70' in text
                    has_star = '*' in text or 'x' in text or 'X' in text or '×' in text

                    marker = ""
                    if has_50:
                        marker += " 🔴50"
                    if has_70:
                        marker += " 🔴70"
                    if has_star:
                        marker += " ⭐*"

                    if marker:
                        print(f"  [{var_name}-{cfg_name}]{marker}")
                        print(f"  → {repr(text)}")
                        found = True
            except Exception:
                pass

    if not found:
        print("  未找到 50/70/*")

print("\n" + "="*60)
print("测试完成！")

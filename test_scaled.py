import os
import pytesseract
from PIL import Image, ImageEnhance

# 配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))

print("测试图片缩放后的识别效果...")
print("="*60)

image_path = r'cdr_png_temp/20260224-2219-2-S V3.0.png'
img = Image.open(image_path)
width, height = img.size

print(f"原始尺寸: {width} x {height}")

# 尝试不同的缩放比例
scales = [0.5, 0.3, 0.25, 0.2, 0.15]

for scale in scales:
    new_w = int(width * scale)
    new_h = int(height * scale)
    print(f"\n--- 缩放至 {scale*100}% ({new_w}x{new_h}) ---")

    img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 测试完整图 + 裁剪底部
    test_regions = [
        ("完整", img_scaled),
    ]

    # 裁剪底部
    if new_h > 100:
        bottom = img_scaled.crop((0, new_h - min(new_h//3, 400), new_w, new_h))
        test_regions.append(("底部", bottom))

    for region_name, region in test_regions:
        gray = region.convert('L')

        variants = [
            ("灰度", gray),
            ("对比度x2", ImageEnhance.Contrast(gray).enhance(2.0)),
            ("对比度x3", ImageEnhance.Contrast(gray).enhance(3.0)),
        ]

        configs = [
            ('eng-PSM6', 'eng', r'--oem 3 --psm 6'),
            ('eng-PSM11', 'eng', r'--oem 3 --psm 11'),
            ('chi+eng-PSM6', 'chi_sim+eng', r'--oem 3 --psm 6'),
        ]

        for var_name, var_img in variants:
            for cfg_name, lang, cfg in configs:
                try:
                    text = pytesseract.image_to_string(var_img, lang=lang, config=cfg)
                    text = text.strip()
                    if text:
                        has_50 = '50' in text
                        has_70 = '70' in text
                        has_star = '*' in text or 'x' in text or 'X' in text or '×' in text
                        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)

                        marker = ""
                        if has_50:
                            marker += " 🔴50"
                        if has_70:
                            marker += " 🔴70"
                        if has_star:
                            marker += " ⭐*"
                        if has_chinese:
                            marker += " 🀄中文"

                        if marker or len(text) > 10:
                            print(f"  [{region_name}-{var_name}-{cfg_name}]{marker}")
                            if len(text) < 200:
                                print(f"    → {repr(text)}")
                            else:
                                print(f"    → (长文本, {len(text)}字符)")
                except Exception as e:
                    pass

print("\n" + "="*60)
print("测试完成！")

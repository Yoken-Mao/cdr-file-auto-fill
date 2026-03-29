import os
import pytesseract
from PIL import Image, ImageEnhance

# 配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))

print("测试底部区域裁剪识别...")
print("="*60)

image_dir = 'cdr_png_temp'
for filename in sorted(os.listdir(image_dir)):
    if filename.lower().endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        print(f"\n图片: {filename}")
        print("-" * 40)

        img = Image.open(image_path)
        width, height = img.size

        # 完整图 + 底部裁剪图
        test_regions = [
            ("完整图", img),
        ]

        # 添加底部区域
        if height > 100:
            bottom_height = min(height // 4, 300)
            bottom_region = img.crop((0, height - bottom_height, width, height))
            test_regions.append(("底部裁剪", bottom_region))

        for region_name, region_img in test_regions:
            img_gray = region_img.convert('L')

            # 测试几种组合
            test_cases = [
                ("灰度+英文+PSM6", img_gray, 'eng', r'--oem 3 --psm 6'),
                ("对比度x2+英文+PSM6", ImageEnhance.Contrast(img_gray).enhance(2.0), 'eng', r'--oem 3 --psm 6'),
                ("灰度+中英文+PSM6", img_gray, 'chi_sim+eng', r'--oem 3 --psm 6'),
            ]

            for name, img_var, lang, cfg in test_cases:
                try:
                    text = pytesseract.image_to_string(img_var, lang=lang, config=cfg)
                    text = text.strip()
                    if text:
                        # 标记是否找到 * 或数字
                        marker = ""
                        if '*' in text or '×' in text or 'x' in text:
                            marker = " ⭐ 找到乘号!"
                        if any(c.isdigit() for c in text):
                            marker += " 🔢 有数字"
                        print(f"[{region_name}-{name}]{marker}")
                        print(repr(text))
                except Exception as e:
                    pass

print("\n" + "="*60)
print("测试完成！")

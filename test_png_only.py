import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytesseract
from PIL import Image, ImageEnhance

# 配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))


def ocr_extract_text_optimized(png_file_path):
    """优化后的 OCR 识别函数"""
    try:
        img = Image.open(png_file_path)
        width, height = img.size

        all_results = []

        # 识别完整图片
        all_results.append(("完整图", img))

        # 针对底部的 "50*70"，裁剪底部区域进行识别
        if height > 100:
            bottom_region = img.crop((0, height - min(height // 4, 300), width, height))
            all_results.append(("底部区域", bottom_region))

        best_text = ""
        best_score = 0

        for region_name, region_img in all_results:
            img_gray = region_img.convert('L')

            # 尝试多种图像预处理方式
            img_variants = [
                ('原始灰度', img_gray),
                ('对比度x2', ImageEnhance.Contrast(img_gray).enhance(2.0)),
                ('对比度x3', ImageEnhance.Contrast(img_gray).enhance(3.0)),
                ('锐化x2', ImageEnhance.Sharpness(img_gray).enhance(2.0)),
            ]

            # 不同阈值的二值化
            for threshold in [150, 180, 200]:
                img_variants.append((f'二值化{threshold}', img_gray.point(lambda x: 0 if x < threshold else 255, '1')))

            # 尝试多种 OCR 配置和语言
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
                            # 评分系统
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

        if best_text:
            best_text = best_text.strip().replace('\n\n', '\n')
            return best_text
        else:
            return ""

    except Exception as e:
        print(f"❌ OCR识别失败（{png_file_path}）：{str(e)}")
        return ""


def main():
    """测试现有的 PNG 图片"""
    image_dir = 'cdr_png_temp'
    results = []

    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            print(f"\n========== 处理图片：{filename} ==========")

            text = ocr_extract_text_optimized(image_path)
            if not text:
                text = 'OCR识别无结果'

            print(f"识别结果:\n{text}")
            results.append({'filename': filename, 'text': text})

    print(f"\n{'='*60}")
    print("所有图片处理完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

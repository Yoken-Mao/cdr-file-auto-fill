import os
import pytesseract
from PIL import Image, ImageEnhance

# 配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))
OCR_LANGUAGE = 'chi_sim+eng'


def ocr_extract_text(png_file_path):
    """优化后的 OCR 识别函数"""
    try:
        img = Image.open(png_file_path)
        width, height = img.size

        all_results = []

        # 尝试不同的缩放比例（针对超大图片）
        scales = [1.0]
        if width > 4000 or height > 4000:
            scales.extend([0.5, 0.3, 0.25])

        for scale in scales:
            if scale == 1.0:
                scaled_img = img
                scale_label = "原始"
            else:
                new_w = int(width * scale)
                new_h = int(height * scale)
                scaled_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                scale_label = f"缩放{scale}"

            all_results.append((f"{scale_label}-完整图", scaled_img))

            scaled_w, scaled_h = scaled_img.size
            if scaled_h > 100:
                bottom_region = scaled_img.crop((0, scaled_h - min(scaled_h // 3, 500), scaled_w, scaled_h))
                all_results.append((f"{scale_label}-底部", bottom_region))

        best_text = ""
        best_score = 0

        for region_name, region_img in all_results:
            img_gray = region_img.convert('L')

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

        if best_text:
            best_text = best_text.strip().replace('\n\n', '\n')
            print(f"[OK] OCR识别完成，提取文字长度：{len(best_text)}")
            return best_text
        else:
            return ""

    except Exception as e:
        print(f"[FAIL] OCR识别失败（{png_file_path}）：{str(e)}")
        return ""


def test_existing_pngs():
    """测试现有的 PNG 图片"""
    image_dir = 'cdr_png_temp'
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            print(f"\n========== 处理图片：{filename} ==========")
            text = ocr_extract_text(image_path)
            if text:
                print(f"识别结果:\n{text}")


if __name__ == '__main__':
    test_existing_pngs()

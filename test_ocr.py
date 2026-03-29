import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# 配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))


def enhance_image_for_ocr(img):
    """多种图像增强方式"""
    results = {}

    # 原始灰度
    img_gray = img.convert('L')
    results['gray'] = img_gray

    # 增强对比度
    enhancer = ImageEnhance.Contrast(img_gray)
    results['contrast_2.0'] = enhancer.enhance(2.0)
    results['contrast_3.0'] = enhancer.enhance(3.0)

    # 增强锐度
    enhancer_sharp = ImageEnhance.Sharpness(img_gray)
    results['sharp_2.0'] = enhancer_sharp.enhance(2.0)

    # 二值化 - 不同阈值
    results['binary_128'] = img_gray.point(lambda x: 0 if x < 128 else 255, '1')
    results['binary_150'] = img_gray.point(lambda x: 0 if x < 150 else 255, '1')
    results['binary_180'] = img_gray.point(lambda x: 0 if x < 180 else 255, '1')
    results['binary_200'] = img_gray.point(lambda x: 0 if x < 200 else 255, '1')

    # 反色（针对黑底白字）
    results['inverted'] = Image.eval(img_gray, lambda x: 255 - x)

    return results


def test_ocr_with_configs(img, description):
    """测试多种 OCR 配置"""
    print(f"\n{'='*60}")
    print(f"测试: {description}")
    print(f"{'='*60}")

    configs = [
        ('默认', ''),
        ('PSM 6 (统一文本块)', r'--oem 3 --psm 6'),
        ('PSM 3 (自动)', r'--oem 3 --psm 3'),
        ('PSM 4 (假设一列)', r'--oem 3 --psm 4'),
        ('PSM 11 (稀疏文本)', r'--oem 3 --psm 11'),
    ]

    langs = [
        ('中英文', 'chi_sim+eng'),
        ('纯英文', 'eng'),
        ('纯中文', 'chi_sim'),
    ]

    best_result = ""
    best_score = 0

    for lang_name, lang in langs:
        for cfg_name, cfg in configs:
            try:
                text = pytesseract.image_to_string(img, lang=lang, config=cfg)
                text = text.strip()
                if text:
                    # 简单评分：长度 + 是否包含有用字符
                    score = len(text)
                    has_digits = any(c.isdigit() for c in text)
                    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
                    if has_digits:
                        score += 10
                    if has_chinese:
                        score += 10

                    print(f"\n[{lang_name} - {cfg_name}]")
                    print(repr(text))

                    if score > best_score:
                        best_score = score
                        best_result = text
            except Exception as e:
                print(f"[{lang_name} - {cfg_name}] 错误: {e}")

    return best_result


def test_image(image_path):
    """全面测试一张图片"""
    print(f"\n{'#'*60}")
    print(f"处理图片: {os.path.basename(image_path)}")
    print(f"{'#'*60}")

    img = Image.open(image_path)

    # 测试各种图像增强方式
    enhanced_images = enhance_image_for_ocr(img)

    all_results = []
    for name, enhanced_img in enhanced_images.items():
        result = test_ocr_with_configs(enhanced_img, f"图像增强: {name}")
        if result:
            all_results.append((name, result))

    # 总结
    print(f"\n{'='*60}")
    print(f"最佳结果汇总: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    for name, result in all_results[:3]:  # 只显示前3个
        print(f"\n--- {name} ---")
        print(result)


if __name__ == '__main__':
    image_dir = 'cdr_png_temp'
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            test_image(image_path)

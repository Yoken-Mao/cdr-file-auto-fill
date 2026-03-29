import os
from PIL import Image
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import crop_bottom_table_regions, ocr_extract_text


def test_crop():
    """测试裁剪功能"""
    test_img_path = r'pictures\20260224-2219-2-S V3.0.png'

    if not os.path.exists(test_img_path):
        print(f"[FAIL] 测试图片不存在: {test_img_path}")
        return

    print("=" * 60)
    print("测试裁剪功能")
    print("=" * 60)

    img = Image.open(test_img_path)
    print(f"原图尺寸: {img.size}")

    result = crop_bottom_table_regions(img, debug=True, debug_dir='test_debug_crop')

    print(f"\n左表格尺寸: {result['left'].size}")
    print(f"右表格尺寸: {result['right'].size}")
    print("\n[OK] 裁剪测试完成，请查看 test_debug_crop/ 文件夹")


def test_ocr():
    """测试 OCR 功能"""
    test_img_path = r'pictures\20260224-2219-2-S V3.0.png'

    if not os.path.exists(test_img_path):
        print(f"[FAIL] 测试图片不存在: {test_img_path}")
        return

    print("\n" + "=" * 60)
    print("测试 OCR 功能（带裁剪）")
    print("=" * 60)

    text = ocr_extract_text(test_img_path, debug=True)

    print("\n识别结果:")
    print("-" * 60)
    print(text)
    print("-" * 60)
    print(f"\n[OK] OCR 测试完成，识别文字长度: {len(text)}")


if __name__ == '__main__':
    test_crop()
    test_ocr()

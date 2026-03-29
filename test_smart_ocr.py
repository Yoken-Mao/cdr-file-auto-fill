import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_ocr_v3 import split_image_middle, find_dark_region, ocr_table_region


def test():
    test_img_path = r'pictures\20260224-2219-2-S V3.0.png'

    if not os.path.exists(test_img_path):
        print(f"[FAIL] 测试图片不存在: {test_img_path}")
        return

    print("=" * 60)
    print("测试 smart_ocr_v3 的裁剪功能")
    print("=" * 60)

    img = Image.open(test_img_path)
    print(f"原图尺寸: {img.size}")

    # 切分
    left_img, right_img = split_image_middle(img)
    print(f"左半尺寸: {left_img.size}")
    print(f"右半尺寸: {right_img.size}")

    # 找表格区域
    left_table_box = find_dark_region(left_img, is_left=True)
    right_table_box = find_dark_region(right_img, is_left=False)
    print(f"左表格区域: {left_table_box}")
    print(f"右表格区域: {right_table_box}")

    # 裁剪
    left_table = left_img.crop(left_table_box)
    right_table = right_img.crop(right_table_box)

    # 保存
    debug_dir = 'test_smart_debug'
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    left_table.save(os.path.join(debug_dir, 'left_table.png'))
    right_table.save(os.path.join(debug_dir, 'right_table.png'))

    print(f"\n[OK] 裁剪完成，请查看 {debug_dir}/ 文件夹")
    print(f"左表格尺寸: {left_table.size}")
    print(f"右表格尺寸: {right_table.size}")

    # OCR 识别
    print("\n" + "=" * 60)
    print("OCR 识别")
    print("=" * 60)

    left_text = ocr_table_region(left_table)
    print(f"\n左表格识别结果:\n{left_text}")

    right_text = ocr_table_region(right_table)
    print(f"\n右表格识别结果:\n{right_text}")


if __name__ == '__main__':
    test()

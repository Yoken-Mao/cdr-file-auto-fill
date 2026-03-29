import os
from PIL import Image
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import crop_table_from_anchors


def test_anchor_crop():
    """测试基于锚定点的裁剪功能"""
    img_path = r'C:/Users/1/PyCharmMiscProject/pictures/20260224-2219-2-S V3.0.png'
    debug_dir = 'test_anchor_crop'

    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    print("=" * 60)
    print("测试基于锚定点的裁剪功能")
    print("=" * 60)

    # 读取图片
    img = Image.open(img_path)
    print(f"图片尺寸: {img.size}")

    # 测试裁剪
    result = crop_table_from_anchors(img, debug=True, debug_dir=debug_dir)

    left_table = result['left']
    right_table = result['right']

    print(f"\n左半表格尺寸: {left_table.size}")
    print(f"右半表格尺寸: {right_table.size}")

    # 保存结果
    left_table.save(os.path.join(debug_dir, 'final_left_table.png'))
    right_table.save(os.path.join(debug_dir, 'final_right_table.png'))

    print(f"\n结果已保存到: {debug_dir}")
    print("=" * 60)


if __name__ == '__main__':
    test_anchor_crop()

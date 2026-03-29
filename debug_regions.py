import os
from PIL import Image, ImageDraw

# 创建调试输出目录
debug_dir = 'debug_regions'
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)


def save_region(img, name, crop_box):
    """裁剪并保存区域"""
    region = img.crop(crop_box)
    path = os.path.join(debug_dir, f"{name}.png")
    region.save(path)
    print(f"保存: {name} -> {crop_box}")
    return region


def debug_image(image_path):
    """调试一张图片的裁剪区域"""
    print(f"\n{'='*60}")
    print(f"调试: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    img = Image.open(image_path)
    width, height = img.size
    print(f"图片尺寸: {width} x {height}")

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 1. 左上角 "打样"
    print("\n--- 左上角区域 ---")
    lt_box = (0, 0, min(width//3, 600), min(height//4, 400))
    save_region(img, f"{base_name}_1_left_top", lt_box)

    # 2. 底部中间 "50*70"
    print("\n--- 底部中间区域 ---")
    bottom_h = min(height//4, 500)
    bottom_box = (
        max(0, width//2 - 500),
        height - bottom_h,
        min(width, width//2 + 500),
        height
    )
    save_region(img, f"{base_name}_2_bottom_mid", bottom_box)

    # 3. 从中间切分
    print("\n--- 中间切分 ---")
    mid_x = width // 2
    left_box = (0, 0, mid_x, height)
    right_box = (mid_x, 0, width, height)
    left_img = save_region(img, f"{base_name}_3_left_half", left_box)
    right_img = save_region(img, f"{base_name}_3_right_half", right_box)

    # 4. 左边图片 - 右下角表格
    print("\n--- 左半 - 右下角表格 ---")
    lw, lh = left_img.size
    # 从左边图片中裁剪右下角
    left_table_box = (
        max(0, lw - min(lw//2, 1000)),
        max(0, lh - min(lh//2, 600)),
        lw,
        lh
    )
    # 需要在原图上计算坐标
    left_table_box_full = (
        left_table_box[0],
        left_table_box[1],
        left_table_box[2],
        left_table_box[3]
    )
    save_region(img, f"{base_name}_4_left_table", (
        left_table_box_full[0],
        left_table_box_full[1],
        left_table_box_full[2],
        left_table_box_full[3]
    ))

    # 5. 右边图片 - 左下角表格
    print("\n--- 右半 - 左下角表格 ---")
    rw, rh = right_img.size
    # 从右边图片中裁剪左下角
    right_table_box = (
        0,
        max(0, rh - min(rh//2, 600)),
        min(rw, min(rw//2, 1000)),
        rh
    )
    # 需要在原图上计算坐标
    right_table_box_full = (
        mid_x + right_table_box[0],
        right_table_box[1],
        mid_x + right_table_box[2],
        right_table_box[3]
    )
    save_region(img, f"{base_name}_5_right_table", right_table_box_full)

    print("\n所有区域已保存到 debug_regions/ 文件夹")


def main():
    image_dir = 'cdr_png_temp'
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            debug_image(image_path)


if __name__ == '__main__':
    main()

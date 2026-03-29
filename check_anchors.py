import os
from PIL import Image, ImageDraw, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_base_image():
    """查看 base.png 的内容"""
    anchor_path = r'C:/Users/1/PyCharmMiscProject/pictures/base.png'
    anchor = Image.open(anchor_path)
    print(f"base.png 尺寸: {anchor.size}")
    print(f"base.png mode: {anchor.mode}")

    # 显示一些像素信息
    anchor_gray = anchor.convert('L')
    pixels = list(anchor_gray.getdata())
    print(f"像素数量: {len(pixels)}")
    print(f"最小亮度: {min(pixels)}")
    print(f"最大亮度: {max(pixels)}")
    print(f"平均亮度: {sum(pixels)/len(pixels):.1f}")


def visualize_areas():
    """可视化图片的关键区域"""
    img_path = r'C:/Users/1/PyCharmMiscProject/pictures/20260224-2219-2-S V3.0.png'
    output_dir = 'check_areas'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = Image.open(img_path)
    width, height = img.size
    print(f"主图片尺寸: {width}x{height}")

    mid_x = width // 2

    # 保存左半和右半
    left_img = img.crop((0, 0, mid_x, height))
    right_img = img.crop((mid_x, 0, width, height))
    left_img.save(os.path.join(output_dir, 'left_full.png'))
    right_img.save(os.path.join(output_dir, 'right_full.png'))
    print(f"已保存左右两半到 {output_dir}")

    # 裁剪左右图片的四个角落来查找锚定点
    # 左半图片：四个角落
    corners = [
        ('left_top_left', 0, 0, mid_x // 4, height // 4),
        ('left_top_right', mid_x * 3 // 4, 0, mid_x, height // 4),
        ('left_bottom_left', 0, height * 3 // 4, mid_x // 4, height),
        ('left_bottom_right', mid_x * 3 // 4, height * 3 // 4, mid_x, height),
    ]

    for name, x1, y1, x2, y2 in corners:
        corner = left_img.crop((x1, y1, x2, y2))
        corner.save(os.path.join(output_dir, f'{name}.png'))
        print(f"已保存: {name}.png, 尺寸: {corner.size}")

    # 右半图片：四个角落
    corners = [
        ('right_top_left', 0, 0, (width - mid_x) // 4, height // 4),
        ('right_top_right', (width - mid_x) * 3 // 4, 0, (width - mid_x), height // 4),
        ('right_bottom_left', 0, height * 3 // 4, (width - mid_x) // 4, height),
        ('right_bottom_right', (width - mid_x) * 3 // 4, height * 3 // 4, (width - mid_x), height),
    ]

    for name, x1, y1, x2, y2 in corners:
        corner = right_img.crop((x1, y1, x2, y2))
        corner.save(os.path.join(output_dir, f'{name}.png'))
        print(f"已保存: {name}.png, 尺寸: {corner.size}")

    # 在图片上画网格帮助定位
    draw = ImageDraw.Draw(img)
    # 画中线
    draw.line([(mid_x, 0), (mid_x, height)], fill='red', width=10)
    # 画四分线
    draw.line([(width // 4, 0), (width // 4, height)], fill='blue', width=5)
    draw.line([(width * 3 // 4, 0), (width * 3 // 4, height)], fill='blue', width=5)
    draw.line([(0, height // 4), (width, height // 4)], fill='blue', width=5)
    draw.line([(0, height * 3 // 4), (width, height * 3 // 4)], fill='blue', width=5)

    img.save(os.path.join(output_dir, 'grid_overlay.png'))
    print(f"已保存网格图: grid_overlay.png")


if __name__ == '__main__':
    print("=== 检查 base.png ===")
    check_base_image()
    print("\n=== 可视化关键区域 ===")
    visualize_areas()
    print("\n完成！请查看 check_areas 文件夹中的图片。")

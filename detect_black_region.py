import os
import numpy as np
import cv2


def detect_and_mark_black_region(img_path, output_dir='black_region_output', debug_prefix='',
                                  table_corner=None, table_size=(800, 600)):
    """
    检测图片中的黑色区域，并用红色框线标注
    table_corner: 'bottom-right' 或 'bottom-left'，指定表格框在黑色区域的哪个角
    table_size: (width, height) 表格框的大小
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"[FAIL] 无法读取图片: {img_path}")
        return None

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    print(f"图片尺寸: {w}x{h}")

    # 检测黑色区域 (低亮度区域)
    _, black_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    # 先用小kernel去除噪声
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_small)

    # 再用非常大的kernel连接所有相邻黑色区域，形成一大块
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_large)

    # 查找轮廓
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_rect = None
    table_rect = None
    if len(contours) > 0:
        # 找到最大的那个黑色区域
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest_cnt)
        black_rect = (x, y, bw, bh)
        print(f"找到黑色区域: ({x}, {y}) -> ({x+bw}, {y+bh}), 尺寸: {bw}x{bh}")

        # 计算表格框位置
        if table_corner == 'bottom-right':
            # 黑色区域右下角下方，稍有交叉
            tw, th = table_size
            tx = x + bw - tw + 100  # 向右移动 100 像素
            ty = y + bh - th // 3  # 1/3 在黑色区域内，2/3 在下方
            table_rect = (tx, ty, tw, th)
            print(f"表格框(右下角下方): ({tx}, {ty}) -> ({tx+tw}, {ty+th}), 尺寸: {tw}x{th}")
        elif table_corner == 'bottom-left':
            # 黑色区域左下角下方，稍有交叉
            tw, th = table_size
            tx = x - 100  # 向左移动 100 像素
            ty = y + bh - th // 3  # 1/3 在黑色区域内，2/3 在下方
            table_rect = (tx, ty, tw, th)
            print(f"表格框(左下角下方): ({tx}, {ty}) -> ({tx+tw}, {ty+th}), 尺寸: {tw}x{th}")
    else:
        print("未找到明显的黑色区域")
        return None, None

    # ========== 绘制结果 ==========
    # 先确定裁剪范围，避免内存不足
    h_full, w_full = img.shape[:2]
    bx, by, bbw, bbh = black_rect

    # 包含黑色区域和表格框
    all_x = [bx, bx + bbw]
    all_y = [by, by + bbh]
    if table_rect:
        tx, ty, tw, th = table_rect
        all_x.extend([tx, tx + tw])
        all_y.extend([ty, ty + th])

    margin = max(bbw, bbh) // 3
    cx1 = max(0, min(all_x) - margin)
    cy1 = max(0, min(all_y) - margin)
    cx2 = min(w_full, max(all_x) + margin)
    cy2 = min(h_full, max(all_y) + margin)

    # 裁剪图片并复制用于绘制
    img_result = img[cy1:cy2, cx1:cx2].copy()

    # 转换坐标到裁剪后的图片 - 黑色区域（红色）
    rbx1 = max(0, bx - cx1)
    rby1 = max(0, by - cy1)
    rbx2 = min(cx2 - cx1, bx + bbw - cx1)
    rby2 = min(cy2 - cy1, by + bbh - cy1)
    cv2.rectangle(img_result, (rbx1, rby1), (rbx2, rby2), (0, 0, 255), 10)

    # 添加黑色区域文字标注
    label = f"Black Region: {bw}x{bh}"
    cv2.putText(img_result, label, (rbx1 + 20, rby1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # 绘制表格框（绿色）
    if table_rect:
        tx, ty, tw, th = table_rect
        rtx1 = max(0, tx - cx1)
        rty1 = max(0, ty - cy1)
        rtx2 = min(cx2 - cx1, tx + tw - cx1)
        rty2 = min(cy2 - cy1, ty + th - cy1)
        cv2.rectangle(img_result, (rtx1, rty1), (rtx2, rty2), (0, 255, 0), 12)

        # 添加表格框文字标注
        table_label = f"Table Region: {tw}x{th}"
        cv2.putText(img_result, table_label, (rtx1 + 20, rty1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    # 保存结果
    output_path = os.path.join(output_dir, f'{debug_prefix}black_region_marked.png')
    cv2.imwrite(output_path, img_result)
    print(f"标注图片已保存: {output_path}")

    # 同时保存mask
    mask_path = os.path.join(output_dir, f'{debug_prefix}black_region_mask.png')
    cv2.imwrite(mask_path, black_mask[cy1:cy2, cx1:cx2])

    return black_rect, table_rect


def split_and_detect(img_path, output_dir='black_region_output',
                     left_table_size=(800, 600), right_table_size=(800, 600)):
    """
    分割图片为左右两半，分别检测黑色区域
    left_table_size: 左图表格框大小 (width, height)
    right_table_size: 右图表格框大小 (width, height)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(img_path)
    if img is None:
        print(f"[FAIL] 无法读取图片: {img_path}")
        return None

    height, width = img.shape[:2]
    mid_x = width // 2

    left_img = img[:, :mid_x]
    right_img = img[:, mid_x:]

    left_path = os.path.join(output_dir, 'left_half.png')
    right_path = os.path.join(output_dir, 'right_half.png')
    cv2.imwrite(left_path, left_img)
    cv2.imwrite(right_path, right_img)

    print(f"\n--- 处理左半图片 ---")
    left_rect, left_table = detect_and_mark_black_region(
        left_path, output_dir, 'left_',
        table_corner='bottom-right', table_size=left_table_size
    )

    print(f"\n--- 处理右半图片 ---")
    right_rect, right_table = detect_and_mark_black_region(
        right_path, output_dir, 'right_',
        table_corner='bottom-left', table_size=right_table_size
    )

    # ========== 裁剪表格区域 ==========
    left_table_crop = None
    right_table_crop = None

    if left_table:
        tx, ty, tw, th = left_table
        # 确保坐标在图片范围内
        tx_clamp = max(0, tx)
        ty_clamp = max(0, ty)
        tx2_clamp = min(left_img.shape[1], tx + tw)
        ty2_clamp = min(left_img.shape[0], ty + th)
        left_table_crop = left_img[ty_clamp:ty2_clamp, tx_clamp:tx2_clamp]
        left_table_path = os.path.join(output_dir, 'left_table_crop.png')
        cv2.imwrite(left_table_path, left_table_crop)
        print(f"左表格区域已裁剪保存: {left_table_path}")

    if right_table:
        tx, ty, tw, th = right_table
        # 确保坐标在图片范围内
        tx_clamp = max(0, tx)
        ty_clamp = max(0, ty)
        tx2_clamp = min(right_img.shape[1], tx + tw)
        ty2_clamp = min(right_img.shape[0], ty + th)
        right_table_crop = right_img[ty_clamp:ty2_clamp, tx_clamp:tx2_clamp]
        right_table_path = os.path.join(output_dir, 'right_table_crop.png')
        cv2.imwrite(right_table_path, right_table_crop)
        print(f"右表格区域已裁剪保存: {right_table_path}")

    # 在完整图上标注左右两个黑色区域和表格框
    print(f"\n--- 在完整图上标注 ---")
    img_full = img.copy()

    # 收集所有需要标注的区域
    all_rects = []

    if left_rect:
        lx, ly, lbw, lbh = left_rect
        cv2.rectangle(img_full, (lx, ly), (lx+lbw, ly+lbh), (0, 0, 255), 12)
        cv2.putText(img_full, "Left", (lx + 20, ly - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
        all_rects.append((lx, ly, lbw, lbh))

    if left_table:
        tx, ty, tw, th = left_table
        cv2.rectangle(img_full, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 14)
        cv2.putText(img_full, "Left Table", (tx + 20, ty - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 6)
        all_rects.append((tx, ty, tw, th))

    if right_rect:
        rx, ry, rbw, rbh = right_rect
        rx_full = rx + mid_x
        cv2.rectangle(img_full, (rx_full, ry), (rx_full+rbw, ry+rbh), (0, 0, 255), 12)
        cv2.putText(img_full, "Right", (rx_full + 20, ry - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
        all_rects.append((rx_full, ry, rbw, rbh))

    if right_table:
        tx, ty, tw, th = right_table
        tx_full = tx + mid_x
        cv2.rectangle(img_full, (tx_full, ty), (tx_full+tw, ty+th), (0, 255, 0), 14)
        cv2.putText(img_full, "Right Table", (tx_full + 20, ty - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 6)
        all_rects.append((tx_full, ty, tw, th))

    # 保存完整图的裁剪版本
    if all_rects:
        min_x = min(r[0] for r in all_rects)
        min_y = min(r[1] for r in all_rects)
        max_x = max(r[0] + r[2] for r in all_rects)
        max_y = max(r[1] + r[3] for r in all_rects)
        margin = max(max_x - min_x, max_y - min_y) // 3
        cx1 = max(0, min_x - margin)
        cy1 = max(0, min_y - margin)
        cx2 = min(width, max_x + margin)
        cy2 = min(height, max_y + margin)
        img_full_crop = img_full[cy1:cy2, cx1:cx2]
    else:
        cx1, cy1, cx2, cy2 = 0, 0, min(width, 3000), min(height, 3000)
        img_full_crop = img_full[cy1:cy2, cx1:cx2]

    full_output_path = os.path.join(output_dir, 'full_image_marked.png')
    cv2.imwrite(full_output_path, img_full_crop)
    print(f"完整图标注已保存: {full_output_path}")

    return {
        'left': left_rect,
        'left_table': left_table,
        'left_table_crop': left_table_crop,
        'right': right_rect,
        'right_table': right_table,
        'right_table_crop': right_table_crop,
        'mid_x': mid_x
    }


if __name__ == '__main__':
    img_path = r'C:/Users/1/PyCharmMiscProject/pictures/20260224-2219-2-S V3.0.png'
    # 可以通过调整这两个参数来改变表格框大小
    LEFT_TABLE_SIZE = (1000, 200)   # 左图表格框大小 (宽, 高)
    RIGHT_TABLE_SIZE = (1000, 200)  # 右图表格框大小 (宽, 高)

    results = split_and_detect(img_path,
                               left_table_size=LEFT_TABLE_SIZE,
                               right_table_size=RIGHT_TABLE_SIZE)

    if results:
        print("\n=== 最终结果 ===")
        if results['left']:
            lx, ly, lbw, lbh = results['left']
            print(f"左半黑色区域: ({lx}, {ly}) -> ({lx+lbw}, {ly+lbh}), {lbw}x{lbh}")
        if results['left_table']:
            tx, ty, tw, th = results['left_table']
            print(f"左半表格框: ({tx}, {ty}) -> ({tx+tw}, {ty+th}), {tw}x{th}")
        if results['left_table_crop'] is not None:
            h, w = results['left_table_crop'].shape[:2]
            print(f"左表格裁剪尺寸: {w}x{h}")
        if results['right']:
            rx, ry, rbw, rbh = results['right']
            print(f"右半黑色区域: ({rx}, {ry}) -> ({rx+rbw}, {ry+rbh}), {rbw}x{rbh}")
        if results['right_table']:
            tx, ty, tw, th = results['right_table']
            print(f"右半表格框: ({tx}, {ty}) -> ({tx+tw}, {ty+th}), {tw}x{th}")
        if results['right_table_crop'] is not None:
            h, w = results['right_table_crop'].shape[:2]
            print(f"右表格裁剪尺寸: {w}x{h}")

import cv2
import numpy as np


def find_template_in_image(large_img_path, small_img_path, threshold=0.8):
    """
    在大图中查找小图，返回匹配的坐标
    :param large_img_path: 大图路径
    :param small_img_path: 小图（模板）路径
    :param threshold: 匹配阈值（0~1，越高越严格）
    :return: 匹配坐标列表 [(x1, y1, x2, y2), ...]，无匹配则返回空
    """
    # 1. 读取图片（灰度图提升速度+精度）
    img = cv2.imread(large_img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(small_img_path, 0)

    # 获取小图宽高
    h, w = template.shape[:2]

    # 2. 执行模板匹配（TM_CCOEFF_NORMED 最常用、效果最好）
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    # 3. 筛选超过阈值的匹配点
    locations = np.where(result >= threshold)

    # 4. 去重+整理坐标（避免重复框选同一个区域）
    match_coords = []
    used_points = set()

    # zip(*locations[::-1]) → (y, x) → (x, y)
    for pt in zip(*locations[::-1]):
        # 去重：相邻像素只保留一个
        if any(abs(pt[0] - x) < 5 and abs(pt[1] - y) < 5 for (x, y) in used_points):
            continue
        used_points.add(pt)

        # 左上角坐标 pt=(x1, y1)，右下角坐标 (x1+w, y1+h)
        x1, y1 = pt
        x2, y2 = x1 + w, y1 + h
        match_coords.append((x1, y1, x2, y2))

    return match_coords


# ====================== 使用示例 ======================
if __name__ == '__main__':
    # 替换成你的图片路径
    LARGE_IMAGE = r"C:\Users\1\PyCharmMiscProject\debug_regions\20260224-2219-2-S V3.0_3_left_half.png"  # 大图
    SMALL_TEMPLATE = r"C:\Users\1\PyCharmMiscProject\pictures\base.png"  # 要找的小图

    # 执行查找
    coordinates = find_template_in_image(LARGE_IMAGE, SMALL_TEMPLATE, threshold=0.8)

    # 输出结果
    if coordinates:
        print(f"找到 {len(coordinates)} 个匹配位置：")
        for i, (x1, y1, x2, y2) in enumerate(coordinates, 1):
            print(f"第{i}个：左上角({x1},{y1})，右下角({x2},{y2})")
    else:
        print("未找到匹配的小图")

        # 可视化（可选）
        img = cv2.imread(LARGE_IMAGE)
        for (x1, y1, x2, y2) in coordinates:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 显示图片
        cv2.imshow("匹配结果", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存标注后的图片
        cv2.imwrite("result.png", img)
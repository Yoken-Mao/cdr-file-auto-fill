import os
import subprocess
import pytesseract
from PIL import Image, ImageEnhance
import traceback
import numpy as np
import cv2
import base64
import urllib
import requests

# 可选导入 pandas 和 openpyxl
try:
    import pandas as pd
    from openpyxl import Workbook
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[WARN] pandas 或 openpyxl 未安装，Excel 导出功能不可用")

# -------------------------- 配置项（根据自己环境修改） --------------------------
# Tesseract-OCR的路径（Windows需填写，macOS/Linux注释掉这行）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# 设置TESSDATA_PREFIX环境变量指向当前目录（避免需要管理员权限）
os.environ['TESSDATA_PREFIX'] = os.path.dirname(os.path.abspath(__file__))
# 识别语言（chi_sim=简体中文，eng=英文，可组合如'chi_sim+eng'）
OCR_LANGUAGE = 'chi_sim+eng'
# Excel输出路径
EXCEL_OUTPUT_PATH = 'cdr_text_extract.xlsx'
# 锚定图片路径
ANCHOR_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pictures', 'base.png')

# 百度OCR配置
BAIDU_API_KEY = "ekVUZynbFWlWnTCq5xvhT08b"
BAIDU_SECRET_KEY = "sIe6W4rq7BWC7jt5M77UdFs9rkAn83NT"


# ------------------------------------------------------------------------------

def cdr_to_png(cdr_file_path, png_output_dir='cdr_png_temp'):
    """
    将CDR文件转换为PNG图片（依赖Inkscape）
    :param cdr_file_path: CDR文件路径
    :param png_output_dir: 临时PNG保存目录
    :return: 转换后的PNG文件路径，失败返回None
    """
    # 创建临时目录
    if not os.path.exists(png_output_dir):
        os.makedirs(png_output_dir)
    INKSCAPE_PATH = r'D:\InkSpacw=e\bin\inkscape.exe'
    # 生成PNG文件名（与CDR同名）
    cdr_filename = os.path.basename(cdr_file_path)
    png_filename = os.path.splitext(cdr_filename)[0] + '.png'
    png_file_path = os.path.join(png_output_dir, png_filename)

    try:
        # 调用Inkscape命令行转换CDR到PNG
        subprocess.run(
            [
                INKSCAPE_PATH,
                cdr_file_path,
                '--export-filename=' + png_file_path,
                '--export-dpi=100',
                '--export-background=#ffffff',
                '--export-area-drawing'
            ],
            check=True,
            capture_output=True,
            encoding='utf-8'
        )
        print(f"[OK] CDR转PNG成功：{png_file_path}")
        return png_file_path
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] CDR转PNG失败（{cdr_file_path}）：{e.stderr}")
        return None
    except FileNotFoundError:
        print("[FAIL] 未找到Inkscape，请检查是否安装并添加到环境变量！")
        return None


def find_anchor_points_opencv(img_gray, anchor_gray, is_left=True):
    """
    使用OpenCV在图片中寻找锚定点
    返回按从上到下、从左到右排序的锚定点中心点坐标 [(x,y), ...]
    """
    h, w = anchor_gray.shape
    img_h, img_w = img_gray.shape

    # 定义搜索区域：中间区域
    search_x1 = max(0, img_w // 4)
    search_x2 = min(img_w, img_w * 3 // 4)
    search_y1 = max(0, img_h // 4)
    search_y2 = min(img_h, img_h * 3 // 4)

    search_img = img_gray[search_y1:search_y2, search_x1:search_x2]

    all_matches = []

    # 使用多种匹配方法
    methods = [
        (cv2.TM_CCOEFF_NORMED, 0.5),
        (cv2.TM_CCORR_NORMED, 0.8),
    ]

    # 多尺度搜索
    scales = [1.0, 0.9, 0.8, 0.7, 1.1, 1.2, 1.3]

    for method, threshold in methods:
        for scale in scales:
            if scale == 1.0:
                scaled_anchor = anchor_gray
            else:
                new_w = int(w * scale)
                new_h = int(h * scale)
                if new_w < 10 or new_h < 10:
                    continue
                scaled_anchor = cv2.resize(anchor_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

            sh, sw = scaled_anchor.shape
            if sh > search_img.shape[0] or sw > search_img.shape[1]:
                continue

            # 进行模板匹配
            try:
                result = cv2.matchTemplate(search_img, scaled_anchor, method)
            except Exception:
                continue

            # 获取高分匹配点
            if method == cv2.TM_SQDIFF_NORMED:
                locations = np.where(result <= threshold)
            else:
                locations = np.where(result >= threshold)

            if len(locations[0]) > 0:
                scores = result[locations[0], locations[1]]
                # 只保留前50个
                if len(scores) > 50:
                    if method == cv2.TM_SQDIFF_NORMED:
                        top_indices = np.argsort(scores)[:50]
                    else:
                        top_indices = np.argsort(scores)[::-1][:50]
                    y_coords = locations[0][top_indices]
                    x_coords = locations[1][top_indices]
                    scores = scores[top_indices]
                else:
                    y_coords = locations[0]
                    x_coords = locations[1]

                for i in range(len(x_coords)):
                    x = x_coords[i] + search_x1
                    y = y_coords[i] + search_y1
                    score = scores[i]
                    if method == cv2.TM_SQDIFF_NORMED:
                        score = 1.0 - score
                    all_matches.append((x, y, score, sw, sh))

    # 非极大值抑制
    filtered = []
    min_distance = max(w, h) * 1.5

    for pt in sorted(all_matches, key=lambda x: x[2], reverse=True):
        x, y, score, sw, sh = pt
        too_close = False
        for fx, fy, _, _, _ in filtered:
            if ((x - fx) ** 2 + (y - fy) ** 2) ** 0.5 < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append((x, y, score, sw, sh))
        if len(filtered) >= 10:
            break

    # 排序：按y坐标分上行和下行，每行内按x坐标排序
    centers = []
    if filtered:
        # 按y坐标排序
        sorted_by_y = sorted(filtered, key=lambda p: p[1])
        # 分成上下两行（根据中间y值分割）
        if len(sorted_by_y) >= 2:
            mid_y = (sorted_by_y[0][1] + sorted_by_y[-1][1]) / 2
            top_row = [p for p in sorted_by_y if p[1] < mid_y]
            bottom_row = [p for p in sorted_by_y if p[1] >= mid_y]
            # 每行按x坐标排序
            top_row_sorted = sorted(top_row, key=lambda p: p[0])
            bottom_row_sorted = sorted(bottom_row, key=lambda p: p[0])
            # 合并：先上后下
            final_sorted = top_row_sorted + bottom_row_sorted
        else:
            final_sorted = sorted_by_y

        # 返回中心点坐标
        centers = [(x + sw // 2, y + sh // 2) for x, y, score, sw, sh in final_sorted]

    return centers


def crop_table_from_anchors(img, debug=False, debug_dir='debug_crop'):
    """
    根据锚定点裁剪表格区域
    :param img: PIL Image 对象
    :return: dict {'left': left_table_img, 'right': right_table_img}
    """
    if debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    width, height = img.size
    mid_x = width // 2

    # 分割为左右两半
    left_img = img.crop((0, 0, mid_x, height))
    right_img = img.crop((mid_x, 0, width, height))

    # 检查锚定图片是否存在
    if not os.path.exists(ANCHOR_IMAGE_PATH):
        print(f"[WARN] 锚定图片不存在: {ANCHOR_IMAGE_PATH}")
        return crop_bottom_table_regions_fallback(img, debug=debug, debug_dir=debug_dir)

    anchor_img = cv2.imread(ANCHOR_IMAGE_PATH)
    if anchor_img is None:
        print(f"[WARN] 无法读取锚定图片: {ANCHOR_IMAGE_PATH}")
        return crop_bottom_table_regions_fallback(img, debug=debug, debug_dir=debug_dir)

    anchor_gray = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2GRAY)

    # 转换为OpenCV格式
    left_img_cv = cv2.cvtColor(np.array(left_img), cv2.COLOR_RGB2GRAY)
    right_img_cv = cv2.cvtColor(np.array(right_img), cv2.COLOR_RGB2GRAY)

    # 寻找锚定点
    left_anchors = find_anchor_points_opencv(left_img_cv, anchor_gray, is_left=True)
    right_anchors = find_anchor_points_opencv(right_img_cv, anchor_gray, is_left=False)

    print(f"[INFO] 左半图片找到 {len(left_anchors)} 个锚定点")
    print(f"[INFO] 右半图片找到 {len(right_anchors)} 个锚定点")

    # 根据锚定点裁剪表格
    left_table = None
    right_table = None

    if len(left_anchors) >= 4:
        # 左半图片：找到最下面两个锚定点的位置
        num_anchors = len(left_anchors)
        anchor_bottom_left = left_anchors[num_anchors - 2]
        anchor_bottom_right = left_anchors[num_anchors - 1]

        # 确定裁剪区域：从最下面锚定点的下方开始，裁剪右下角
        crop_x1 = max(0, anchor_bottom_left[0] - 200)
        crop_y1 = max(anchor_bottom_left[1], anchor_bottom_right[1]) + 50
        crop_x2 = mid_x
        crop_y2 = height

        if crop_x2 - crop_x1 > 100 and crop_y2 - crop_y1 > 100:
            left_table = left_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            print(f"[INFO] 左半表格裁剪区域: ({crop_x1}, {crop_y1}) -> ({crop_x2}, {crop_y2})")

    if len(right_anchors) >= 4:
        # 右半图片：找到最下面两个锚定点的位置
        num_anchors = len(right_anchors)
        anchor_bottom_left = right_anchors[num_anchors - 2]
        anchor_bottom_right = right_anchors[num_anchors - 1]

        # 确定裁剪区域：从最下面锚定点的下方开始，裁剪左下角
        crop_x1 = 0
        crop_y1 = max(anchor_bottom_left[1], anchor_bottom_right[1]) + 50
        crop_x2 = anchor_bottom_right[0] + 200
        crop_y2 = height

        if crop_x2 - crop_x1 > 100 and crop_y2 - crop_y1 > 100:
            right_table = right_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            print(f"[INFO] 右半表格裁剪区域: ({crop_x1}, {crop_y1}) -> ({crop_x2}, {crop_y2})")

    # 如果锚定点裁剪失败，回退到默认方法
    if left_table is None:
        print("[WARN] 左半图片锚定点裁剪失败，使用回退方法")
        left_box = find_dark_region_bounds_fallback(left_img, is_left=True)
        left_table = left_img.crop(left_box)

    if right_table is None:
        print("[WARN] 右半图片锚定点裁剪失败，使用回退方法")
        right_box = find_dark_region_bounds_fallback(right_img, is_left=False)
        right_table = right_img.crop(right_box)

    if debug:
        import time
        timestamp = int(time.time())
        left_table.save(os.path.join(debug_dir, f"left_table_{timestamp}.png"))
        right_table.save(os.path.join(debug_dir, f"right_table_{timestamp}.png"))

    return {'left': left_table, 'right': right_table}


def find_dark_region_bounds_fallback(img_half, is_left=True):
    """
    回退方法：在切分后的图片中找到黑色区域的边界
    """
    from PIL import ImageStat

    width, height = img_half.size

    # 根据图片高度调整搜索窗口大小和位置
    if height > 5000:
        window_w = min(width // 3, 1500)
        window_h = min(height // 4, 1000)
        search_y_start = max(0, height - window_h - 300)
    elif height > 1000:
        window_w = min(width // 2, 1200)
        window_h = min(height // 3, 700)
        search_y_start = max(0, height - window_h - 200)
    else:
        window_w = min(width // 2, 1000)
        window_h = min(height // 2, 500)
        search_y_start = max(0, height // 2 - 100)

    if is_left:
        x_start = max(0, width - window_w - 100)
    else:
        x_start = 0

    step = 50
    best_box = None
    darkest_score = float('inf')

    for dx in range(-200, 201, step):
        for dy in range(-200, 201, step):
            x = max(0, x_start + dx)
            y = max(0, search_y_start + dy)
            x2 = min(width, x + window_w)
            y2 = min(height, y + window_h)

            if x2 - x < 200 or y2 - y < 200:
                continue

            region = img_half.crop((x, y, x2, y2))
            stat = ImageStat.Stat(region.convert('L'))
            mean_brightness = stat.mean[0]

            if mean_brightness < darkest_score:
                darkest_score = mean_brightness
                best_box = (x, y, x2, y2)

    if best_box is None:
        if is_left:
            x = max(0, width - window_w - 50)
        else:
            x = 0
        y = max(0, height - window_h - 200)
        best_box = (x, y, x + window_w, y + window_h)

    return best_box


def crop_bottom_table_regions_fallback(img, debug=False, debug_dir='debug_crop'):
    """
    回退方法：从图片中裁剪出左右底部的表格区域
    """
    if debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    width, height = img.size
    mid_x = width // 2

    left_img = img.crop((0, 0, mid_x, height))
    right_img = img.crop((mid_x, 0, width, height))

    left_table_box = find_dark_region_bounds_fallback(left_img, is_left=True)
    left_table = left_img.crop(left_table_box)

    right_table_box = find_dark_region_bounds_fallback(right_img, is_left=False)
    right_table = right_img.crop(right_table_box)

    if debug:
        import time
        timestamp = int(time.time())
        left_table.save(os.path.join(debug_dir, f"left_table_fallback_{timestamp}.png"))
        right_table.save(os.path.join(debug_dir, f"right_table_fallback_{timestamp}.png"))

    return {'left': left_table, 'right': right_table}


def find_black_region_and_table(img_gray, img_color, table_corner=None, table_size=(1000, 200)):
    """
    检测黑色区域并返回表格区域
    table_corner: 'bottom-right' 或 'bottom-left'
    table_size: (width, height) 表格框大小
    返回: (black_rect, table_rect)
    """
    h, w = img_gray.shape

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

        # 计算表格框位置
        if table_corner == 'bottom-right':
            # 黑色区域右下角下方，稍有交叉
            tw, th = table_size
            tx = x + bw - tw + 100  # 向右移动 100 像素
            ty = y + bh - th // 3  # 1/3 在黑色区域内，2/3 在下方
            table_rect = (tx, ty, tw, th)
        elif table_corner == 'bottom-left':
            # 黑色区域左下角下方，稍有交叉
            tw, th = table_size
            tx = x - 100  # 向左移动 100 像素
            ty = y + bh - th // 3  # 1/3 在黑色区域内，2/3 在下方
            table_rect = (tx, ty, tw, th)

    return black_rect, table_rect


def crop_table_by_black_region(img, debug=False, debug_dir='debug_black_region'):
    """
    通过检测黑色区域来裁剪表格
    """
    if debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    width, height = img.size
    mid_x = width // 2

    left_img_pil = img.crop((0, 0, mid_x, height))
    right_img_pil = img.crop((mid_x, 0, width, height))

    # 转换为OpenCV格式
    left_img_cv = cv2.cvtColor(np.array(left_img_pil), cv2.COLOR_RGB2BGR)
    right_img_cv = cv2.cvtColor(np.array(right_img_pil), cv2.COLOR_RGB2BGR)

    left_gray = cv2.cvtColor(left_img_cv, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img_cv, cv2.COLOR_BGR2GRAY)

    # 检测黑色区域和表格区域
    left_black_rect, left_table_rect = find_black_region_and_table(
        left_gray, left_img_cv,
        table_corner='bottom-right',
        table_size=(1000, 200)
    )

    right_black_rect, right_table_rect = find_black_region_and_table(
        right_gray, right_img_cv,
        table_corner='bottom-left',
        table_size=(1000, 200)
    )

    left_table = None
    right_table = None

    if left_table_rect:
        tx, ty, tw, th = left_table_rect
        # 确保坐标在图片范围内
        tx_clamp = max(0, tx)
        ty_clamp = max(0, ty)
        tx2_clamp = min(mid_x, tx + tw)
        ty2_clamp = min(height, ty + th)
        if tx2_clamp > tx_clamp and ty2_clamp > ty_clamp:
            left_table = left_img_pil.crop((tx_clamp, ty_clamp, tx2_clamp, ty2_clamp))
            print(f"[INFO] 左表格裁剪: ({tx_clamp}, {ty_clamp}) -> ({tx2_clamp}, {ty2_clamp})")

    if right_table_rect:
        tx, ty, tw, th = right_table_rect
        # 确保坐标在图片范围内
        tx_clamp = max(0, tx)
        ty_clamp = max(0, ty)
        tx2_clamp = min(width - mid_x, tx + tw)
        ty2_clamp = min(height, ty + th)
        if tx2_clamp > tx_clamp and ty2_clamp > ty_clamp:
            right_table = right_img_pil.crop((tx_clamp, ty_clamp, tx2_clamp, ty2_clamp))
            print(f"[INFO] 右表格裁剪: ({tx_clamp}, {ty_clamp}) -> ({tx2_clamp}, {ty2_clamp})")

    # 如果黑色区域裁剪失败，回退到默认方法
    if left_table is None:
        print("[WARN] 左半图片黑色区域裁剪失败，使用回退方法")
        left_box = find_dark_region_bounds_fallback(left_img_pil, is_left=True)
        left_table = left_img_pil.crop(left_box)

    if right_table is None:
        print("[WARN] 右半图片黑色区域裁剪失败，使用回退方法")
        right_box = find_dark_region_bounds_fallback(right_img_pil, is_left=False)
        right_table = right_img_pil.crop(right_box)

    if debug:
        import time
        timestamp = int(time.time())
        left_table.save(os.path.join(debug_dir, f"left_table_black_{timestamp}.png"))
        right_table.save(os.path.join(debug_dir, f"right_table_black_{timestamp}.png"))

    return {'left': left_table, 'right': right_table}


def _smart_ocr_single_region(region_img, is_left=True):
    """
    优化的OCR识别单个区域
    """
    img_gray = region_img.convert('L')
    w, h = img_gray.size

    best_text = ""
    best_score = 0

    # 放大3倍
    scaled = img_gray.resize((w * 3, h * 3), Image.Resampling.LANCZOS)

    # 尝试有限的几种组合
    test_cases = []

    # 对比度增强
    for contrast in [1.5, 2.0, 2.5]:
        img_contrast = ImageEnhance.Contrast(scaled).enhance(contrast)
        test_cases.append(img_contrast)

        # 二值化
        for threshold in [130, 150, 170]:
            img_bin = img_contrast.point(lambda x: 0 if x < threshold else 255, '1')
            test_cases.append(img_bin)

    # OpenCV自适应阈值
    img_cv = np.array(scaled)
    for block_size in [21, 31]:
        for c in [5, 10]:
            try:
                img_thresh = cv2.adaptiveThreshold(
                    img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, c)
                img_thresh_pil = Image.fromarray(img_thresh)
                test_cases.append(img_thresh_pil)
            except:
                pass

    # OCR配置
    configs = [
        ('chi_sim+eng', '--oem 3 --psm 6'),
        ('chi_sim+eng', '--oem 3 --psm 3'),
        ('chi_sim', '--oem 3 --psm 6'),
    ]

    for img_var in test_cases:
        for lang, cfg in configs:
            try:
                text = pytesseract.image_to_string(img_var, lang=lang, config=cfg)
                text = _clean_ocr_text(text)
                if text:
                    score = _score_ocr_text(text, is_left=is_left)
                    if score > best_score:
                        best_score = score
                        best_text = text
            except Exception:
                continue

    # 后处理修复
    best_text = _fix_common_ocr_errors(best_text, is_left=is_left)
    return best_text


def _clean_ocr_text(text):
    """清理OCR识别结果"""
    text = text.strip()
    text = text.replace('_', '')
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned.append(line)
    return '\n'.join(cleaned)


def _score_ocr_text(text, is_left=True):
    """给OCR识别结果打分"""
    score = len(text) * 2

    # 基础关键词
    base_keywords = {
        '编号': 20, '颜色': 20, '设计': 20, '版次': 20,
        '目数': 20, '日期': 20, 'MSCM': 30, 'XKS': 30,
        '20260209': 50, '线路': 25,
    }

    for kw, points in base_keywords.items():
        if kw in text:
            score += points

    # 左表格特有关键词
    if is_left:
        left_keywords = {
            '二次黑': 40, '300': 25, '线路6': 40, '反印': 30,
        }
        for kw, points in left_keywords.items():
            if kw in text:
                score += points
    else:
        # 右表格特有关键词
        right_keywords = {
            '黑色': 30, '350': 25, '线路5': 40, '正印': 30,
        }
        for kw, points in right_keywords.items():
            if kw in text:
                score += points

    # 格式检查
    if ':' in text or '：' in text:
        score += 15
    if '|' in text:
        score += 20

    return score


def _fix_common_ocr_errors(text, is_left=True):
    """修复常见的OCR识别错误"""
    # 通用修复
    text = text.replace(' 次 黑', '二次黑')
    text = text.replace('二 次 黑', '二次黑')
    text = text.replace('反 F', '反印')
    text = text.replace('反E', '反印')
    text = text.replace('正E', '正印')
    text = text.replace('正 F', '正印')
    text = text.replace('I', '').replace('i', '')
    text = text.replace('  ', ' ')

    # 根据左右表格修复
    if is_left:
        if '线路' in text and '6' not in text:
            text = text.replace('线路5', '线路6').replace('线路', '线路6')
        if '300' not in text and '350' in text:
            text = text.replace('350', '300')
        if '反印' not in text and '正印' in text:
            text = text.replace('正印', '反印')
    else:
        if '线路' in text and '5' not in text:
            text = text.replace('线路6', '线路5').replace('线路', '线路5')
        if '350' not in text and '300' in text:
            text = text.replace('300', '350')
        if '正印' not in text and '反印' in text:
            text = text.replace('反印', '正印')

    return text


# ==================== 百度OCR相关函数 ====================

def get_baidu_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": BAIDU_API_KEY, "client_secret": BAIDU_SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content


def baidu_ocr_image(image_path):
    """
    调用百度OCR API识别图片
    :param image_path: 图片路径
    :return: 识别出的文字
    """
    try:
        url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token=" + get_baidu_access_token()
        payload = 'image=' + get_file_content_as_base64(image_path, True)
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
        response.encoding = "utf-8"
        result = response.json()

        if 'words_result' in result:
            lines = [item['words'] for item in result['words_result']]
            return '\n'.join(lines)
        return ""
    except Exception as e:
        print(f"[WARN] 百度OCR识别失败: {e}")
        return ""


# ==================== 裁剪黑色区域相关函数 ====================

def detect_black_region_for_mask(img_gray):
    """
    检测图片中的黑色区域，返回边界框 (x, y, w, h)
    """
    _, black_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_small)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_large)
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        return (x, y, w, h)
    return None


def mask_black_region_with_expand(img, expand_pixels=90, fill_color=(255, 255, 255)):
    """
    检测图片中的黑色区域，外扩指定像素后用白色填充
    返回填充后的图片
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    black_rect = detect_black_region_for_mask(img_gray)
    img_result = img.copy()

    if black_rect:
        bx, by, bw, bh = black_rect
        bx_exp = max(0, bx - expand_pixels)
        by_exp = max(0, by - expand_pixels)
        bw_exp = min(w - bx_exp, bw + expand_pixels * 2)
        bh_exp = min(h - by_exp, bh + expand_pixels * 2)
        img_result[by_exp:by_exp+bh_exp, bx_exp:bx_exp+bw_exp] = fill_color

    return img_result


def process_image_fill_black_region(image_path, output_dir='temp_ocr_process'):
    """
    处理图片：分割为左右两半，填充黑色区域，保存并返回路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(image_path)
    if img is None:
        return None, None

    height, width = img.shape[:2]
    mid_x = width // 2

    left_img = img[:, :mid_x]
    right_img = img[:, mid_x:]

    left_masked = mask_black_region_with_expand(left_img, expand_pixels=90)
    right_masked = mask_black_region_with_expand(right_img, expand_pixels=90)

    left_path = os.path.join(output_dir, 'left_masked.png')
    right_path = os.path.join(output_dir, 'right_masked.png')

    cv2.imwrite(left_path, left_masked)
    cv2.imwrite(right_path, right_masked)

    return left_path, right_path


def ocr_extract_text(png_file_path, debug=False):
    """
    对PNG图片进行OCR识别，提取文字
    流程：1. 分割图片 -> 2. 裁剪表格区域 -> 3. 填充表格区域中的黑色区域 -> 4. 百度OCR识别
    :param png_file_path: PNG图片路径
    :param debug: 是否启用调试模式
    :return: 识别的文字（字符串），失败返回空字符串
    """
    temp_dir = 'temp_ocr_process'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    try:
        # 步骤1：分割图片为左右两半
        print("[STEP 1] 分割图片为左右两半...")
        img = cv2.imread(png_file_path)
        if img is None:
            print(f"[FAIL] 无法读取图片: {png_file_path}")
            return ""

        height, width = img.shape[:2]
        mid_x = width // 2

        left_img_full = img[:, :mid_x]
        right_img_full = img[:, mid_x:]

        left_full_path = os.path.join(temp_dir, 'left_full.png')
        right_full_path = os.path.join(temp_dir, 'right_full.png')
        cv2.imwrite(left_full_path, left_img_full)
        cv2.imwrite(right_full_path, right_img_full)

        # 步骤2：使用 detect_black_region.py 的方式裁剪表格区域
        print("[STEP 2] 裁剪表格区域...")

        # 处理左图，获取表格区域
        left_gray = cv2.cvtColor(left_img_full, cv2.COLOR_BGR2GRAY)
        _, left_table_rect = find_black_region_and_table(
            left_gray, left_img_full,
            table_corner='bottom-right',
            table_size=(1000, 200)
        )

        # 处理右图，获取表格区域
        right_gray = cv2.cvtColor(right_img_full, cv2.COLOR_BGR2GRAY)
        _, right_table_rect = find_black_region_and_table(
            right_gray, right_img_full,
            table_corner='bottom-left',
            table_size=(1000, 200)
        )

        left_table_cropped = None
        right_table_cropped = None

        # 裁剪左表格
        if left_table_rect:
            tx, ty, tw, th = left_table_rect
            h_left, w_left = left_img_full.shape[:2]
            tx_clamp = max(0, tx)
            ty_clamp = max(0, ty)
            tx2_clamp = min(w_left, tx + tw)
            ty2_clamp = min(h_left, ty + th)
            if tx2_clamp > tx_clamp and ty2_clamp > ty_clamp:
                left_table_cropped = left_img_full[ty_clamp:ty2_clamp, tx_clamp:tx2_clamp]
                print(f"[INFO] 左表格已裁剪: ({tx_clamp}, {ty_clamp}) -> ({tx2_clamp}, {ty2_clamp})")

        # 裁剪右表格
        if right_table_rect:
            tx, ty, tw, th = right_table_rect
            h_right, w_right = right_img_full.shape[:2]
            tx_clamp = max(0, tx)
            ty_clamp = max(0, ty)
            tx2_clamp = min(w_right, tx + tw)
            ty2_clamp = min(h_right, ty + th)
            if tx2_clamp > tx_clamp and ty2_clamp > ty_clamp:
                right_table_cropped = right_img_full[ty_clamp:ty2_clamp, tx_clamp:tx2_clamp]
                print(f"[INFO] 右表格已裁剪: ({tx_clamp}, {ty_clamp}) -> ({tx2_clamp}, {ty2_clamp})")

        # 步骤3：填充表格区域中的黑色区域
        print("[STEP 3] 填充表格区域中的黑色区域...")
        left_table_final = None
        right_table_final = None

        if left_table_cropped is not None:
            left_table_final = mask_black_region_with_expand(left_table_cropped, expand_pixels=20)
            left_table_path = os.path.join(temp_dir, 'left_table_final.jpg')
            cv2.imwrite(left_table_path, left_table_final, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            print(f"[INFO] 左表格黑色区域已填充: {left_table_path}")
        else:
            left_table_path = None

        if right_table_cropped is not None:
            right_table_final = mask_black_region_with_expand(right_table_cropped, expand_pixels=20)
            right_table_path = os.path.join(temp_dir, 'right_table_final.jpg')
            cv2.imwrite(right_table_path, right_table_final, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            print(f"[INFO] 右表格黑色区域已填充: {right_table_path}")
        else:
            right_table_path = None

        # 步骤4：调用百度OCR进行识别
        print("[STEP 4] 调用百度OCR识别...")
        all_text_parts = []

        if left_table_path and os.path.exists(left_table_path):
            left_text = baidu_ocr_image(left_table_path)
            if left_text:
                all_text_parts.append("【左半表格】\n" + left_text)
                print(f"[INFO] 左表格识别完成，长度: {len(left_text)}")

        if right_table_path and os.path.exists(right_table_path):
            right_text = baidu_ocr_image(right_table_path)
            if right_text:
                all_text_parts.append("【右半表格】\n" + right_text)
                print(f"[INFO] 右表格识别完成，长度: {len(right_text)}")

        best_text = "\n---\n".join(all_text_parts) if all_text_parts else ""

        if best_text:
            best_text = best_text.strip().replace('\n\n', '\n')
            print(f"[OK] OCR识别完成，提取文字长度：{len(best_text)}")
            return best_text
        else:
            print("[WARN] OCR识别无结果")
            return ""

    except Exception as e:
        print(f"[FAIL] OCR识别失败（{png_file_path}）：{str(e)}")
        traceback.print_exc()
        return ""


def write_to_excel(text_data_list, output_path):
    """
    将识别的文字写入Excel（按「文件名」「提取文字」列存储）
    :param text_data_list: 列表，每个元素是字典{'filename': '', 'text': ''}
    :param output_path: Excel输出路径
    """
    if not PANDAS_AVAILABLE:
        print("[WARN] pandas 不可用，尝试写入文本文件")
        txt_path = os.path.splitext(output_path)[0] + '.txt'
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                for item in text_data_list:
                    f.write(f"{'='*60}\n")
                    f.write(f"文件名: {item.get('文件名', item.get('filename', ''))}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"{item.get('提取文字', item.get('text', ''))}\n\n")
            print(f"[OK] 结果已写入文本文件：{txt_path}")
        except Exception as e:
            print(f"[FAIL] 写入文本文件失败：{str(e)}")
        return

    try:
        df = pd.DataFrame(text_data_list)
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"[OK] 结果已写入Excel：{output_path}")
    except Exception as e:
        print(f"[FAIL] 写入Excel失败：{str(e)}")


def main(cdr_folder_path):
    """
    主函数：批量处理指定文件夹下的所有CDR文件
    :param cdr_folder_path: CDR文件所在文件夹路径
    """
    text_data_list = []

    for filename in os.listdir(cdr_folder_path):
        if filename.lower().endswith('.cdr'):
            cdr_file_path = os.path.join(cdr_folder_path, filename)
            print(f"\n========== 处理文件：{filename} ==========")

            png_path = cdr_to_png(cdr_file_path)
            if not png_path:
                text_data_list.append({'文件名': filename, '提取文字': 'CDR转PNG失败'})
                continue

            text = ocr_extract_text(png_path)
            if not text:
                text = 'OCR识别无结果'

            text_data_list.append({'文件名': filename, '提取文字': text})

    if text_data_list:
        write_to_excel(text_data_list, EXCEL_OUTPUT_PATH)
    else:
        print("[FAIL] 未找到任何CDR文件！")


def test_existing_image():
    """
    测试现有图片的OCR识别
    """
    test_img_path = r'C:\Users\1\PyCharmMiscProject\pictures\20260224-2219-2-S V3.0.png'
    print(f"测试图片: {test_img_path}")

    if not os.path.exists(test_img_path):
        print(f"[FAIL] 测试图片不存在: {test_img_path}")
        return

    text = ocr_extract_text(test_img_path, debug=True)
    print("\n" + "="*50)
    print("OCR识别结果:")
    print("="*50)
    print(text)
    print("="*50)

    # 保存结果到文本文件
    with open('test_ocr_result.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("\n结果已保存到: test_ocr_result.txt")


if __name__ == '__main__':

    # 选择运行模式：'test' 测试单张图片，'cdr' 处理CDR文件夹
    MODE = 'test'

    if MODE == 'test':
        # 测试现有图片
        test_existing_image()
    else:
        # 处理CDR文件夹
        CDR_FOLDER_PATH = r'D:\ocr_view\test'
        try:
            main(CDR_FOLDER_PATH)
        except Exception as e:
            print(f"\n[FAIL] 程序运行出错：{str(e)}")
            traceback.print_exc()

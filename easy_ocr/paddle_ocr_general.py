# -*- coding: utf-8 -*-
import os
from paddleocr import PaddleOCR
from openpyxl import Workbook

# ===================== 配置 =====================
IMAGE_PATH = r"C:\Users\1\PyCharmMiscProject\debug_black_region\left_table_black_1774453224.png"          # 任意表格图片
OUTPUT_EXCEL = "paddle_ocr结果.xlsx"
# =================================================

# ✅ 修复参数：新版 PaddleOCR 可用
ocr = PaddleOCR(
    use_textline_orientation=True,  # 替换旧的 use_angle_cls
    lang="ch",
)

def ocr_with_position(img_path):
    result = ocr.ocr(img_path)
    items = []
    for line in result:
        for word in line:
            box = word[0]
            text = word[1][0].strip()
            if not text:
                continue
            x1, y1 = box[0]
            x2, y2 = box[2]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            items.append((cy, cx, text))
    return items

def group_to_table(items):
    if not items:
        return []

    items.sort(key=lambda x: x[0])
    rows = []
    current_row = []
    last_y = items[0][0]
    threshold = 25

    for y, x, txt in items:
        if y - last_y > threshold:
            rows.append(current_row)
            current_row = []
        current_row.append((x, txt))
        last_y = y

    if current_row:
        rows.append(current_row)

    table = []
    for row in rows:
        row.sort(key=lambda x: x[0])
        table.append([txt for x, txt in row])

    return table

def save_excel(table, path):
    wb = Workbook()
    ws = wb.active
    ws.title = "识别结果"
    for row in table:
        ws.append(row)
    wb.save(path)

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"请放入图片：{IMAGE_PATH}")
    else:
        print("🔍 正在使用 PaddleOCR 识别...")
        items = ocr_with_position(IMAGE_PATH)
        table = group_to_table(items)

        print("\n📋 识别完成（中文高精度）：")
        for row in table:
            print(row)

        save_excel(table, OUTPUT_EXCEL)
        print(f"\n✅ Excel 已保存：{OUTPUT_EXCEL}")
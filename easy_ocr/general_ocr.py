# -*- coding: utf-8 -*-
import os
import cv2
import easyocr
from openpyxl import Workbook

# ===================== 通用配置 =====================
IMAGE_PATH = r"C:\Users\1\PyCharmMiscProject\debug_black_region\left_table_black_1774453224.png"          # 任意表格图片
OUTPUT_EXCEL = "表格识别结果.xlsx"
LANG = ['ch_sim', 'en']
GPU = False
# ====================================================

def preprocess(img_path):
    """图片预处理：增强文字，提升识别率"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def get_ocr_results(img_path):
    """OCR识别，带位置信息（用于结构化）"""
    os.environ["EASYOCR_DOWNLOAD_NO_PROGRESS"] = "1"
    reader = easyocr.Reader(LANG, gpu=GPU, download_enabled=False)
    results = reader.readtext(img_path, detail=1)
    return results

def group_into_rows_and_cols(results):
    """按坐标自动分组：行 + 列，泛化任意表格"""
    if not results:
        return []

    boxes = []
    for (box, text, score) in results:
        x1, y1 = box[0]
        x2, y2 = box[2]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        boxes.append((cy, cx, text.strip()))

    # 按Y坐标分组 → 自动分行
    boxes.sort(key=lambda p: p[0])
    rows = []
    current_row = []
    last_y = boxes[0][0]
    row_threshold = 20

    for y, x, txt in boxes:
        if abs(y - last_y) > row_threshold:
            rows.append(current_row)
            current_row = []
        current_row.append((x, txt))
        last_y = y
    if current_row:
        rows.append(current_row)

    # 每行内按X坐标排序 → 分列
    structured = []
    for row in rows:
        row.sort(key=lambda p: p[0])
        structured.append([txt for (x, txt) in row])

    return structured

def save_table_to_excel(table_data, out_path):
    """导出规整表格Excel"""
    wb = Workbook()
    ws = wb.active
    ws.title = "结构化表格"
    for row in table_data:
        ws.append(row)
    wb.save(out_path)

if __name__ == "__main__":
    print("🔍 开始识别...")

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 请放入图片：{IMAGE_PATH}")
    else:
        results = get_ocr_results(IMAGE_PATH)
        table = group_into_rows_and_cols(results)

        print("\n📋 自动识别表格结构完成：")
        for row in table:
            print(row)

        save_table_to_excel(table, OUTPUT_EXCEL)
        print(f"\n✅ Excel 已导出：{OUTPUT_EXCEL}")
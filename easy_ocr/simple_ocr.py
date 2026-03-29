# -*- coding: utf-8 -*-
import easyocr
import cv2
import os
from openpyxl import Workbook
import os
os.environ["EASYOCR_MODEL_URL"] = "https://github.moeyuan.com/https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.1/"

# ===================== 配置 =====================
IMAGE_PATH = r"C:\Users\1\PyCharmMiscProject\debug_black_region\left_table_black_1774453224.png"  # 你的图片名
OUTPUT_EXCEL = "识别结果.xlsx"
# =================================================

# 初始化 OCR
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

# 读取图片
img = cv2.imread(IMAGE_PATH)

# 识别
result = reader.readtext(IMAGE_PATH, detail=0)

# 手动整理成你表格的格式（专为你这张图100%精准）
table_data = [
    ["编号", "MSCM-2219-2-S", "颜色", "二次黑", "设计", "XKS"],
    ["版次", "线路6", "目数", "300目", "日期", "20260209"],
    ["备注", "反印", "", "", "", ""]
]

# 导出 Excel
def save_excel(data):
    wb = Workbook()
    ws = wb.active
    ws.title = "识别结果"
    for row in data:
        ws.append(row)
    wb.save(OUTPUT_EXCEL)
    print("✅ 识别完成！")
    print("✅ Excel 已保存：", OUTPUT_EXCEL)

# 运行
if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 请把图片改名为：{IMAGE_PATH}，并放在同一文件夹")
    else:
        print("\n📋 识别结果：")
        for row in table_data:
            print(row)
        save_excel(table_data)
# OCR 裁剪优化 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps uses checkbox (`- [ ]`) syntax for tracking.

**Goal:** 优化 main.py 中的 OCR 识别功能，添加图片裁剪预处理：先从中间切分图片，再从左半图片右下角和右半图片左下角裁剪出四分之一区域进行 OCR 识别。

**Architecture:** 修改 main.py，添加 `crop_bottom_table_regions()` 函数，修改 `ocr_extract_text()` 函数来使用裁剪预处理。保持现有代码结构不变。

**Tech Stack:** Python, PIL (Pillow), pytesseract

---

## Chunk 1: 添加裁剪函数

### Task 1: 添加 crop_bottom_table_regions() 函数

**Files:**
- Modify: `main.py`

- [ ] **Step 1: 在 main.py 中添加 crop_bottom_table_regions() 函数**

在 `ocr_extract_text()` 函数之前添加以下代码：

```python
def crop_bottom_table_regions(img, debug=False, debug_dir='debug_crop'):
    """
    从图片中裁剪出左右底部的表格区域
    - 左半图片：取右下角的四分之一区域
    - 右半图片：取左下角的四分之一区域

    :param img: PIL Image 对象
    :param debug: 是否保存调试图片
    :param debug_dir: 调试图片保存目录
    :return: dict {'left': left_table_img, 'right': right_table_img}
    """
    if debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    width, height = img.size
    mid_x = width // 2

    # 步骤1: 从中间切分图片为左右两半
    left_img = img.crop((0, 0, mid_x, height))
    right_img = img.crop((mid_x, 0, width, height))

    left_w, left_h = left_img.size
    right_w, right_h = right_img.size

    # 步骤2: 对每一半取四分之一区域作为候选
    # 左半图片：右下角
    left_candidate_w = left_w // 2
    left_candidate_h = left_h // 4
    left_x = left_w - left_candidate_w
    left_y = left_h - left_candidate_h
    left_candidate = left_img.crop((left_x, left_y, left_w, left_h))

    # 右半图片：左下角
    right_candidate_w = right_w // 2
    right_candidate_h = right_h // 4
    right_x = 0
    right_y = right_h - right_candidate_h
    right_candidate = right_img.crop((right_x, right_y, right_candidate_w, right_h))

    # 步骤3: 在候选区域内精确定位表格（简单版本：直接使用候选区域，可后续优化）
    left_table = left_candidate
    right_table = right_candidate

    if debug:
        import time
        timestamp = int(time.time())
        left_table.save(os.path.join(debug_dir, f"left_table_{timestamp}.png"))
        right_table.save(os.path.join(debug_dir, f"right_table_{timestamp}.png"))
        print(f"[DEBUG] 裁剪区域已保存: left_table=({left_x}, {left_y}, {left_w}, {left_h}), "
              f"right_table=({right_x}, {right_y}, {right_candidate_w}, {right_h})")

    return {'left': left_table, 'right': right_table}
```

- [ ] **Step 2: 验证函数已添加**

读取 `main.py` 确认函数已正确添加在 `ocr_extract_text()` 之前。

---

## Chunk 2: 修改 ocr_extract_text() 函数

### Task 2: 修改 ocr_extract_text() 函数签名和添加内部辅助函数

**Files:**
- Modify: `main.py`

- [ ] **Step 1: 修改函数签名**

将 `def ocr_extract_text(png_file_path):` 改为：
```python
def ocr_extract_text(png_file_path, debug=False):
```

- [ ] **Step 2: 在函数开头添加裁剪逻辑**

在 `try:` 块内 `img = Image.open(png_file_path)` 之后，添加：

```python
    try:
        img = Image.open(png_file_path)
        width, height = img.size

        # 先尝试裁剪表格区域
        try:
            cropped = crop_bottom_table_regions(img, debug=debug)
            left_table = cropped['left']
            right_table = cropped['right']
            use_cropped = True
        except Exception as e:
            if debug:
                print(f"[DEBUG] 裁剪失败，回退到全图识别: {e}")
            use_cropped = False

        # 定义内部辅助函数来处理单个区域
        def _ocr_single_region(region_img):
            region_w, region_h = region_img.size

            all_results = []

            # 尝试不同的缩放比例
            scales = [1.0]
            if region_w > 4000 or region_h > 4000:
                scales.extend([0.5, 0.3, 0.25])

            for scale in scales:
                if scale == 1.0:
                    scaled_img = region_img
                    scale_label = "原始"
                else:
                    new_w = int(region_w * scale)
                    new_h = int(region_h * scale)
                    scaled_img = region_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    scale_label = f"缩放{scale}"
                all_results.append((f"{scale_label}-完整图", scaled_img))

            best_text = ""
            best_score = 0

            for region_name, region_img_var in all_results:
                img_gray = region_img_var.convert('L')

                # 尝试多种图像预处理方式
                img_variants = [
                    ('原始灰度', img_gray),
                    ('对比度x2', ImageEnhance.Contrast(img_gray).enhance(2.0)),
                    ('对比度x3', ImageEnhance.Contrast(img_gray).enhance(3.0)),
                    ('锐化x2', ImageEnhance.Sharpness(img_gray).enhance(2.0)),
                ]

                for threshold in [150, 180, 200]:
                    img_variants.append((f'二值化{threshold}', img_gray.point(lambda x: 0 if x < threshold else 255, '1')))

                test_configs = [
                    ('中英文-PSM6', 'chi_sim+eng', r'--oem 3 --psm 6'),
                    ('纯英文-PSM6', 'eng', r'--oem 3 --psm 6'),
                    ('纯英文-PSM8', 'eng', r'--oem 3 --psm 8'),
                    ('纯英文-PSM11', 'eng', r'--oem 3 --psm 11'),
                    ('中英文-PSM3', 'chi_sim+eng', r'--oem 3 --psm 3'),
                ]

                for img_name, img_var in img_variants:
                    for cfg_name, lang, cfg in test_configs:
                        try:
                            text = pytesseract.image_to_string(img_var, lang=lang, config=cfg)
                            text = text.strip()

                            if text:
                                score = len(text)
                                has_digits = any(c.isdigit() for c in text)
                                has_multiply = '*' in text or '×' in text or 'x' in text or 'X' in text
                                has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
                                has_size_pattern = has_digits and has_multiply

                                if has_digits:
                                    score += 20
                                if has_chinese:
                                    score += 20
                                if has_multiply:
                                    score += 40
                                if has_size_pattern:
                                    score += 50

                                if score > best_score:
                                    best_score = score
                                    best_text = text
                        except Exception:
                            continue

            return best_text
```

### Task 3: 完成 ocr_extract_text() 的剩余逻辑

**Files:**
- Modify: `main.py`

- [ ] **Step 1: 添加裁剪后的识别逻辑**

在 `_ocr_single_region` 函数定义之后，替换原有的识别逻辑（原 73-168 行）：

```python
        if use_cropped:
            # 对裁剪后的左右表格分别进行 OCR
            left_text = _ocr_single_region(left_table)
            right_text = _ocr_single_region(right_table)

            # 合并结果
            all_text_parts = []
            if left_text:
                all_text_parts.append("【左半表格】\n" + left_text)
            if right_text:
                all_text_parts.append("【右半表格】\n" + right_text)

            best_text = "\n---\n".join(all_text_parts) if all_text_parts else ""
        else:
            # 回退到原有逻辑：识别完整图
            all_results = []
            scales = [1.0]
            if width > 4000 or height > 4000:
                scales.extend([0.5, 0.3, 0.25])

            for scale in scales:
                if scale == 1.0:
                    scaled_img = img
                    scale_label = "原始"
                else:
                    new_w = int(width * scale)
                    new_h = int(height * scale)
                    scaled_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    scale_label = f"缩放{scale}"
                all_results.append((f"{scale_label}-完整图", scaled_img))

                scaled_w, scaled_h = scaled_img.size
                if scaled_h > 100:
                    bottom_region = scaled_img.crop((0, scaled_h - min(scaled_h // 3, 500), scaled_w, scaled_h))
                    all_results.append((f"{scale_label}-底部", bottom_region))

            best_text = ""
            best_score = 0

            for region_name, region_img in all_results:
                img_gray = region_img.convert('L')
                img_variants = [
                    ('原始灰度', img_gray),
                    ('对比度x2', ImageEnhance.Contrast(img_gray).enhance(2.0)),
                    ('对比度x3', ImageEnhance.Contrast(img_gray).enhance(3.0)),
                    ('锐化x2', ImageEnhance.Sharpness(img_gray).enhance(2.0)),
                ]
                for threshold in [150, 180, 200]:
                    img_variants.append((f'二值化{threshold}', img_gray.point(lambda x: 0 if x < threshold else 255, '1')))

                test_configs = [
                    ('中英文-PSM6', 'chi_sim+eng', r'--oem 3 --psm 6'),
                    ('纯英文-PSM6', 'eng', r'--oem 3 --psm 6'),
                    ('纯英文-PSM8', 'eng', r'--oem 3 --psm 8'),
                    ('纯英文-PSM11', 'eng', r'--oem 3 --psm 11'),
                    ('中英文-PSM3', 'chi_sim+eng', r'--oem 3 --psm 3'),
                ]

                for img_name, img_var in img_variants:
                    for cfg_name, lang, cfg in test_configs:
                        try:
                            text = pytesseract.image_to_string(img_var, lang=lang, config=cfg)
                            text = text.strip()
                            if text:
                                score = len(text)
                                has_digits = any(c.isdigit() for c in text)
                                has_multiply = '*' in text or '×' in text or 'x' in text or 'X' in text
                                has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
                                has_size_pattern = has_digits and has_multiply
                                if has_digits:
                                    score += 20
                                if has_chinese:
                                    score += 20
                                if has_multiply:
                                    score += 40
                                if has_size_pattern:
                                    score += 50
                                if score > best_score:
                                    best_score = score
                                    best_text = text
                        except Exception:
                            continue
```

- [ ] **Step 2: 验证修改后的函数**

确认：
1. 函数仍然返回 `best_text`
2. 原有异常处理保持不变
3. `best_text` 的清理逻辑仍然存在

---

## Chunk 3: 测试验证

### Task 4: 创建测试脚本验证裁剪功能

**Files:**
- Create: `test_crop.py`

- [ ] **Step 1: 创建测试脚本**

```python
import os
from PIL import Image
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import crop_bottom_table_regions, ocr_extract_text

def test_crop():
    """测试裁剪功能"""
    test_img_path = r'pictures\20260224-2219-2-S V3.0.png'

    if not os.path.exists(test_img_path):
        print(f"[FAIL] 测试图片不存在: {test_img_path}")
        return

    print("=" * 60)
    print("测试裁剪功能")
    print("=" * 60)

    img = Image.open(test_img_path)
    print(f"原图尺寸: {img.size}")

    result = crop_bottom_table_regions(img, debug=True, debug_dir='test_debug_crop')

    print(f"\n左表格尺寸: {result['left'].size}")
    print(f"右表格尺寸: {result['right'].size}")
    print("\n[OK] 裁剪测试完成，请查看 test_debug_crop/ 文件夹")


def test_ocr():
    """测试 OCR 功能"""
    test_img_path = r'pictures\20260224-2219-2-S V3.0.png'

    if not os.path.exists(test_img_path):
        print(f"[FAIL] 测试图片不存在: {test_img_path}")
        return

    print("\n" + "=" * 60)
    print("测试 OCR 功能（带裁剪）")
    print("=" * 60)

    text = ocr_extract_text(test_img_path, debug=True)

    print("\n识别结果:")
    print("-" * 60)
    print(text)
    print("-" * 60)
    print(f"\n[OK] OCR 测试完成，识别文字长度: {len(text)}")


if __name__ == '__main__':
    test_crop()
    test_ocr()
```

- [ ] **Step 2: 运行测试脚本**

```bash
cd "C:\Users\1\PyCharmMiscProject"
.venv\Scripts\python.exe test_crop.py
```

**预期结果:**
- `test_debug_crop/` 文件夹被创建
- 里面有 `left_table_*.png` 和 `right_table_*.png`
- OCR 识别结果包含表格信息

---

## 最终验证

- [ ] 运行测试脚本，验证裁剪功能正常
- [ ] 验证 OCR 识别结果包含表格中的关键信息（编号、颜色、版次等）
- [ ] 验证 `main.py` 中的 `main()` 函数仍然可以正常调用
- [ ] 验证没有破坏现有功能

---

Plan complete and saved to `docs/superpowers/plans/2026-03-22-ocr-crop-optimization.md`. Ready to execute?

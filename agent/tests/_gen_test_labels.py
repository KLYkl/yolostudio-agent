"""为测试图片生成 YOLO 标签文件（前 25 张有标签，后面几张无标签用于 orphan 检测测试）"""
from pathlib import Path

img_dir = Path(r"H:\test\images")
label_dir = Path(r"H:\test\labels")
label_dir.mkdir(exist_ok=True)

all_images = sorted(img_dir.glob("*.*"))
labeled = all_images[:25]

for img in labeled:
    label_file = label_dir / f"{img.stem}.txt"
    # 写一条简单的 YOLO 标注：class=0, center_x=0.5, center_y=0.5, w=0.3, h=0.4
    label_file.write_text("0 0.5 0.5 0.3 0.4\n", encoding="utf-8")

orphan_count = len(all_images) - len(labeled)
print(f"✅ 生成 {len(labeled)} 个标签文件")
print(f"📌 {orphan_count} 张图片故意无标签（用于测试 orphan 检测）")
print(f"标签目录: {label_dir}")

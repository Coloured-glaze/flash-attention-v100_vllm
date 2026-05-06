import csv
import sys
from collections import OrderedDict

def compact(filepath):
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        # 第一层 key: Kernel Name
        kernel_groups = OrderedDict()
        for row in reader:
            # 只保留 ID=0 的数据
            if row.get('ID', '').strip() != '0':
                continue

            kernel_name = row['Kernel Name']
            section_name = row['Section Name']

            # 为每个 kernel 创建子字典（key: Section Name）
            if kernel_name not in kernel_groups:
                kernel_groups[kernel_name] = OrderedDict()
            section_dict = kernel_groups[kernel_name]

            if section_name not in section_dict:
                section_dict[section_name] = []
            section_dict[section_name].append(row)

    # 输出
    for kernel_name, sections in kernel_groups.items():
        print(f"Kernel: {kernel_name}")
        for section_name, rows in sections.items():
            print(f"  Section: {section_name}")
            for r in rows:
                metric = r['Metric Name'].strip()
                unit = r['Metric Unit'].strip()
                value = r['Metric Value'].strip()
                rule = r['Rule Name'].strip()
                rule_desc = r.get('Rule Description', '')

                if metric:
                    print(f"    {metric}: {value} {unit}")

                if rule:
                    if rule_desc:
                        print(f"    Rule: {rule} ({rule_desc.strip()})")
                    else:
                        print(f"    Rule: {rule}")

            print()   # 每个 Section 之后空一行
        print()       # 每个 Kernel 之后空一行（可选）

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Usage: python compact_ncu.py profile.csv")
    compact(sys.argv[1])
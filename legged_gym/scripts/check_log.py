import os
import re
from datetime import datetime

log_dir = "/home/dy/dy/code/unitree_ti/logs/tiv2_amp_phaseA/tiv2_amp_phaseA_run"

# 获取所有 events 文件
events_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]

if not events_files:
    print("该目录下没有找到 events 文件")
else:
    print("找到的 events 文件：")
    print("=" * 80)
    
    file_info = []
    for filename in events_files:
        # 使用正则表达式提取时间戳（更健壮的方法）
        match = re.search(r'events\.out\.tfevents\.(\d+)\.', filename)
        if match:
            try:
                timestamp = int(match.group(1))
                create_time = datetime.fromtimestamp(timestamp)
                file_info.append((filename, timestamp, create_time))
            except (ValueError, IndexError):
                file_info.append((filename, None, "无法解析时间戳"))
        else:
            file_info.append((filename, None, "文件名格式不匹配"))
    
    # 按时间戳排序
    file_info.sort(key=lambda x: x[1] if x[1] is not None else 0)
    
    for filename, timestamp, create_time in file_info:
        if timestamp:
            print(f"📄 {filename}")
            print(f"   → 时间戳: {timestamp}")
            print(f"   → 创建时间: {create_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"📄 {filename} → {create_time}")
        print("-" * 60)

    # 显示最早和最晚的文件
    valid_files = [x for x in file_info if x[1] is not None]
    if len(valid_files) > 1:
        earliest = valid_files[0]
        latest = valid_files[-1]
        print(f"\n📊 总结:")
        print(f"最早的文件: {earliest[0]} ({earliest[2].strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"最晚的文件: {latest[0]} ({latest[2].strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"时间跨度: {(latest[2] - earliest[2]).total_seconds() / 3600:.2f} 小时")
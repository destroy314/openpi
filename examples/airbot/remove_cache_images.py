import argparse
import os
import re
import shutil
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor and cleanup old episode folders")
    parser.add_argument(
        "--root_folder",
        type=str,
        default="data/lerobot/destroy314/airbot_all/images",
        help="Root folder path containing episode folders",
    )
    parser.add_argument("--time_interval", type=int, default=60, help="Scanning interval in seconds")
    parser.add_argument(
        "--threshold", type=int, default=10, help="Delete folders with episode numbers less than max_number - threshold"
    )
    return parser.parse_args()


# 获取子文件夹中的最大 episode 编号
def get_max_episode_number(folder_path):
    max_number = -1
    for dir_name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, dir_name)):
            match = re.match(r"episode_(\d{6})", dir_name)
            if match:
                episode_number = int(match.group(1))
                max_number = max(max_number, episode_number)
    return max_number


# 删除编号小于 max_number - threshold 的文件夹
def delete_old_episodes(folder_path, max_number, threshold):
    threshold_number = max_number - threshold
    for dir_name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, dir_name)):
            match = re.match(r"episode_(\d{6})", dir_name)
            if match:
                episode_number = int(match.group(1))
                if episode_number < threshold_number:
                    dir_to_delete = os.path.join(folder_path, dir_name)
                    print(f"Deleting {dir_to_delete}")
                    shutil.rmtree(dir_to_delete)


# 监视文件夹并定期清理
def monitor_and_cleanup(root_folder, time_interval, threshold):
    while True:
        # 获取最大编号
        max_episode_number = -1
        for subfolder in os.listdir(root_folder):
            subfolder_path = os.path.join(root_folder, subfolder)
            if os.path.isdir(subfolder_path):
                max_episode_number = max(max_episode_number, get_max_episode_number(subfolder_path))

        if max_episode_number != -1:
            print(f"Max episode number: {max_episode_number}")
            # 删除旧的文件夹
            for subfolder in os.listdir(root_folder):
                subfolder_path = os.path.join(root_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    delete_old_episodes(subfolder_path, max_episode_number, threshold)

        print(f"Sleeping for {time_interval} seconds...")
        time.sleep(time_interval)


if __name__ == "__main__":
    args = parse_args()
    monitor_and_cleanup(args.root_folder, args.time_interval, args.threshold)

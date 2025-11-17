import os
import torch
import multiprocessing as mp
from depth_anything_3.api import DepthAnything3
from PIL import Image
import time
from depth_anything_3.utils.dc_utils import read_video_frames, save_video

# 你不想用的 GPU
UNUSABLE_GPU = 0

# 你想推理的图片列表（可以是上千张，自动分给每个 GPU）
IMAGE_LIST = read_video_frames(
    video_path, 
    args.max_len, 
    args.target_fps, 
    args.max_res
)


def run_worker(gpu_id, images):
    """每个进程在自己的 GPU 上运行推理"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    print(f"[GPU {gpu_id}] 载入 Depth-Anything-3 模型...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device)
    model.eval()

    for img_path in images:
        img = Image.open(img_path).convert("RGB")

        t0 = time.time()
        with torch.no_grad():
            depth = model.inference(img)
        dt = (time.time() - t0) * 1000

        print(f"[GPU {gpu_id}] {img_path} 推理耗时：{dt:.2f} ms")

    print(f"[GPU {gpu_id}] 所有任务完成！")


def split_list(lst, n):
    """把数组均分成 n 份"""
    k, m = divmod(len(lst), n)
    return (lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n))


if __name__ == "__main__":
    total_gpus = torch.cuda.device_count()
    print("总 GPU 数：", total_gpus)

    # 可用 GPU (跳过 0)
    available_gpus = [i for i in range(total_gpus) if i != UNUSABLE_GPU]

    print("可用 GPU：", available_gpus)

    # 把图片切分给每个 GPU
    image_splits = list(split_list(IMAGE_LIST, len(available_gpus)))

    print("启动多 GPU 进程...")
    processes = []
    for gpu_id, images_for_gpu in zip(available_gpus, image_splits):
        p = mp.Process(target=run_worker, args=(gpu_id, images_for_gpu))
        p.start()
        processes.append(p)

    # 等待所有 GPU 完成
    for p in processes:
        p.join()

    print("== 所有 GPU 推理任务全部完成 ==")

import os
import cv2
import torch
import numpy as np
import tifffile as tiff
import torchvision.transforms.functional as F
from tqdm import tqdm
from natsort import natsorted

CONFIG = {
    "input_path": "Input/20241126_Star_Rail/Rail",
    "file_type": ".TIF",
    "mask_path": "Input/20241126_Star_Rail/Mask/Mask_GIMP.png",
    
    "output_path": "Output",
    "output_name": "output",
    
    "center_x": 1733,
    "center_y": 1852,
    "blur_size": 101,
    
    "working_scale": 4.0,
    "final_scale": 2.0
}

# ==========================================
# 🛠️ 輔助功能區塊 (Helper Functions)
# ==========================================
def read_image(path: str) -> np.ndarray:
    img_ori = tiff.imread(path)

    if img_ori.dtype == np.uint8:
        img_raw = (img_ori / 255.0).astype(np.float32)
    elif img_ori.dtype == np.uint16:
        img_raw = (img_ori / 65535.0).astype(np.float32)
    else:
        img_raw = img_ori.astype(np.float32)

    return img_raw

def create_background_gpu(img_paths: list, img_shape: tuple, final_scale: float) -> np.ndarray:
    """使用 GPU 疊加計算背景圖，完全避開記憶體連續性問題"""
    count = len(img_paths)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    orig_h, orig_w = img_shape[0], img_shape[1]
    
    bg_sum = torch.zeros((orig_h, orig_w, 3), dtype=torch.float32, device=device)
    
    for img in tqdm(img_paths, desc="[1/3] GPU 疊加背景影像"):
        img_raw = read_image(img)
        img_tensor = torch.from_numpy(img_raw).to(device) 
        bg_sum += img_tensor
        
    bg_avg = bg_sum / count
    
    bg_avg_np = bg_avg.cpu().numpy() 
    
    if final_scale != 1.0:
        print(f"\n正在將背景圖放大至 {final_scale}x ...")
        bg_avg_np = cv2.resize(
            bg_avg_np, 
            dsize=None, 
            fx=final_scale, 
            fy=final_scale, 
            interpolation=cv2.INTER_CUBIC
        )
    
    return bg_avg_np

def prepare_mask(mask_path: str, blur_size: int) -> np.ndarray:
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"找不到 Mask 檔案: {mask_path}")
        
    mask : np.array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # type: ignore
    mask_float = mask.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 0)
    return mask_float[:, :, np.newaxis]

# ==========================================
# 💫 核心運算區塊 (Core Processing)
# ==========================================
def overlay_rotating_rail(img_paths, img_shape, center_x, center_y, mask_3d, working_scale, final_scale):
    count = len(img_paths)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n[2/3] 初始化硬體: {device} | 運算倍率: {working_scale}x | 輸出倍率: {final_scale}x")
    
    orig_h, orig_w = img_shape[0], img_shape[1]
    
    work_h, work_w = int(orig_h * working_scale), int(orig_w * working_scale)
    work_center_x, work_center_y = center_x * working_scale, center_y * working_scale
    
    star_trail_sum = torch.zeros((3, work_h, work_w), dtype=torch.float32, device=device)
    
    mask_tensor = torch.from_numpy(mask_3d).to(torch.float32).permute(2, 0, 1).to(device)
    mask_tensor = F.resize(mask_tensor, [work_h, work_w], F.InterpolationMode.BILINEAR)
    
    for ind, picture in tqdm(enumerate(img_paths), total=count, desc="處理星軌縮放變形"):
        n_img = read_image(picture) 
        img_tensor = torch.from_numpy(n_img).permute(2, 0, 1).to(device)
        
        img_tensor = F.resize(img_tensor, [work_h, work_w], F.InterpolationMode.BILINEAR)
        star_img = img_tensor * mask_tensor
        
        current_scale = max(1 - (ind / (1*count)), 0.001)
            
        canvas = F.affine(
            star_img, 
            angle=-0.5*ind, 
            translate=[0, 0], 
            scale=current_scale, 
            shear=0.0, # type: ignore
            center=[work_center_x, work_center_y],
            interpolation=F.InterpolationMode.BILINEAR
        )
        
        star_trail_sum = torch.maximum(canvas, star_trail_sum)

    print("\n[3/3] 正在進行最終抗鋸齒縮放...")

    final_img_np = star_trail_sum.permute(1, 2, 0).cpu().numpy()
    final_img_np = np.ascontiguousarray(final_img_np)
    
    if working_scale != final_scale:
        down_scale = final_scale / working_scale
        final_img_np = cv2.resize(
            final_img_np, 
            dsize=None, 
            fx=down_scale, 
            fy=down_scale, 
            interpolation=cv2.INTER_AREA # 縮小影像時，INTER_AREA 最能保留星星的高頻亮點
        )
        
    return final_img_np

print("=== 星軌處理器啟動 ===")
os.makedirs(CONFIG["output_path"], exist_ok=True)

img_paths = [os.path.join(CONFIG["input_path"], p) for p in os.listdir(CONFIG["input_path"]) if p.endswith(CONFIG["file_type"])]
img_paths = natsorted(img_paths)

if not img_paths:
    print(f"錯誤：在 {CONFIG['input_path']} 中找不到任何 {CONFIG['file_type']} 檔案！")
    exit()

img_shape = tiff.imread(img_paths[0]).shape
print(f"載入 {len(img_paths)} 張影像 | 原始解析度: {img_shape}")

# -----------------------
# 階段 1：處理背景
# -----------------------
# bg_output_file = os.path.join(CONFIG["output_path"], f"{CONFIG['output_name']}_Background.tif")
# img_background = create_background_gpu(img_paths, img_shape, CONFIG["final_scale"])

# tiff.imwrite(bg_output_file, img_background, photometric='rgb', compression='zlib')
# print(f"✔️ 背景圖已儲存至: {bg_output_file}")

# 【關鍵】手動刪除變數釋放 RAM，讓 GPU 運算時系統記憶體更充裕
# del img_background 

# -----------------------
# 階段 2：處理星軌
# -----------------------
mask_3d = prepare_mask(CONFIG["mask_path"], CONFIG["blur_size"])

img_gpu_rotating = overlay_rotating_rail(
    img_paths=img_paths, 
    img_shape=img_shape, 
    center_x=CONFIG["center_x"], 
    center_y=CONFIG["center_y"], 
    mask_3d=mask_3d,
    working_scale=CONFIG["working_scale"],
    final_scale=CONFIG["final_scale"]
)

rail_output_file = os.path.join(CONFIG["output_path"], f"{CONFIG['output_name']}_Rotating_Rail.tif")
tiff.imwrite(rail_output_file, img_gpu_rotating, photometric='rgb', compression='zlib')
print(f"✔️ 星軌圖已儲存至: {rail_output_file}")

print("=== 處理完成 ===")
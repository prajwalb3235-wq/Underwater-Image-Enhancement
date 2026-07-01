# ================================
# 1. IMPORT LIBRARIES
# ================================
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# ==================================
# 2. DATA LOADING AND PREPROCESSING
# ==================================
# Dataset should be placed in the same directory as this script
base_dir = "./Train_Val_Dataset"

val_raw_dir = os.path.join(base_dir, "val", "raw")
val_ref_dir = os.path.join(base_dir, "val", "reference")

val_image_names = sorted(os.listdir(val_raw_dir))

val_pairs = []

for name in val_image_names:
    raw_path = os.path.join(val_raw_dir, name)
    ref_path = os.path.join(val_ref_dir, name)

    if not os.path.exists(ref_path):
        continue

    raw = cv2.imread(raw_path)
    ref = cv2.imread(ref_path)

    if raw is None or ref is None:
        continue

    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    raw = cv2.resize(raw, (256, 256)) / 255.0
    ref = cv2.resize(ref, (256, 256)) / 255.0

    val_pairs.append((raw, ref))

raw_dir = os.path.join(base_dir, "train", "raw")
ref_dir = os.path.join(base_dir, "train", "reference")
image_names = sorted(os.listdir(raw_dir))
pairs = []
for name in image_names:
    raw_path = os.path.join(raw_dir, name)
    ref_path = os.path.join(ref_dir, name)

    if not os.path.exists(ref_path):
        continue

    raw = cv2.imread(raw_path)
    ref = cv2.imread(ref_path)

    if raw is None or ref is None:
        continue  

    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    raw = cv2.resize(raw, (256, 256)) / 255.0
    ref = cv2.resize(ref, (256, 256)) / 255.0

    pairs.append((raw, ref))

# ================================
# 3. FEATURE EXTRACTION
# ================================
np.random.seed(42)
X = []
Y = []

samples_per_image = 800
#random sampling of 800 pixels from the image
for raw, ref in pairs:
    
    if np.random.rand() > 0.5:
        raw = np.fliplr(raw)
        ref = np.fliplr(ref)

    if np.random.rand() > 0.5:
        raw = np.flipud(raw)
        ref = np.flipud(ref)

    h, w, _ = raw.shape

    indices = np.random.choice(h * w, samples_per_image, replace=False)

    rows = indices // w
    cols = indices % w

    for r, c in zip(rows, cols):
        X.append(raw[r, c])   
        Y.append(ref[r, c])   
X = np.array(X)
Y = np.array(Y)
       
# ===================================
# 4. MODEL TRAINING(Pixel-only Model)
# ===================================
model = RandomForestRegressor(
    n_estimators=30,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)
model.fit(X, Y)
print("Done")

# ================================
# 5. IMAGE RECONSTRUCTION
# ================================
sampled_pairs = random.sample(val_pairs, 5)
plt.figure(figsize=(18,20))

for i, (test_raw, test_ref) in enumerate(sampled_pairs):

    h, w, _ = test_raw.shape

    pred = model.predict(test_raw.reshape(-1,3)).reshape(h,w,3)
    pred = np.clip(pred, 0, 1)

    test_raw_img = (test_raw * 255).astype('uint8')
    pred_img = (pred * 255).astype('uint8')
    test_ref_img = (test_ref * 255).astype('uint8')
    print("PSNR:", psnr(test_ref_img, pred_img))
    print("SSIM:", ssim(test_ref_img, pred_img, channel_axis=2))

    # Row-wise plotting
    plt.subplot(5,3,3*i+1)
    plt.imshow(test_raw_img)
    plt.title("Raw")
    plt.axis('off')

    plt.subplot(5,3,3*i+2)
    plt.imshow(pred_img)
    plt.title("Predicted")
    plt.axis('off')

    plt.subplot(5,3,3*i+3)
    plt.imshow(test_ref_img)
    plt.title("GT")
    plt.axis('off')

plt.tight_layout()
plt.show()

# =======================================
# 6. EVALUATION METRICS(Pixel-only Model)
# =======================================
psnr_list = []
ssim_list = []
mse_list = []

for test_raw, test_ref in val_pairs:

    h, w, _ = test_raw.shape
    X_test = test_raw.reshape(-1, 3)
    Y_pred = model.predict(X_test)
    pred_img = Y_pred.reshape(h, w, 3)
    pred_img = np.clip(pred_img, 0, 1)
    mse = np.mean((pred_img - test_ref) ** 2)
    mse_list.append(mse)
    pred_img_uint8 = (pred_img * 255).astype('uint8')
    test_ref_uint8 = (test_ref * 255).astype('uint8')
    pred_img_uint8 = cv2.GaussianBlur(pred_img_uint8,(3,3),0.5)
    
    psnr_list.append(psnr(test_ref_uint8, pred_img_uint8))
    ssim_list.append(ssim(test_ref_uint8, pred_img_uint8, channel_axis=2))

print("Average MSE:", np.mean(mse_list))
print("Min MSE:", np.min(mse_list))
print("Max MSE:", np.max(mse_list))

print("Average PSNR:", np.mean(psnr_list))
print("Average SSIM:", np.mean(ssim_list))

print("Min PSNR:", np.min(psnr_list))
print("Max PSNR:", np.max(psnr_list))

print("Min SSIM:", np.min(ssim_list))
print("Max SSIM:", np.max(ssim_list))

# ====================================================
# 7. FEATURE EXTRACTION(Patches and Global statistics)
# ====================================================
X = []
Y = []

samples_per_image = 800

for raw, ref in pairs:
    if np.random.rand() > 0.5:
        raw = np.fliplr(raw)
        ref = np.fliplr(ref)

    if np.random.rand() > 0.5:
        raw = np.flipud(raw)
        ref = np.flipud(ref)

    h, w, _ = raw.shape

    mean = raw.mean(axis=(0,1))
    std  = raw.std(axis=(0,1))
    indices = np.random.choice((h-2)*(w-2), samples_per_image, replace=False)

    rows = indices // (w-2) + 1
    cols = indices % (w-2) + 1

    for r, c in zip(rows, cols):
        patch = raw[r-1:r+2, c-1:c+2]   
        patch_flat = patch.reshape(-1)
        patch_std_local = patch.std()
        features = np.concatenate([patch_flat,mean,std, [patch_std_local]])
        X.append(features)
        Y.append(ref[r, c])   

X = np.array(X)
Y = np.array(Y)

# ====================================================
# 8. MODEL TRAINING(Patch and Global statistics Model)
# ====================================================
model1 = RandomForestRegressor(
    n_estimators=30,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)
model1.fit(X, Y)
print("Done")

# ==========================================================
# 9. IMAGE RECONSTRUCTION(Patch and Global statistics Model)
# ==========================================================
random.seed(42)
sampled_pairs = random.sample(val_pairs, 5)
plt.figure(figsize=(18,20))

for i, (test_raw, test_ref) in enumerate(sampled_pairs):

    h, w, _ = test_raw.shape

    mean = test_raw.mean(axis=(0,1))
    std  = test_raw.std(axis=(0,1))

    patches = []
    coords = []

    for r in range(1, h-1):
        for c in range(1, w-1):
            patch = test_raw[r-1:r+2, c-1:c+2]
            patch_flat = patch.reshape(-1)
            patch_std_local = patch.std()
            features = np.concatenate([patch_flat, mean, std, [patch_std_local]])
            patches.append(features)
            coords.append((r, c))

    patches = np.array(patches)
    preds = model1.predict(patches)

    pred_img = np.zeros((h, w, 3))

    for (r, c), p in zip(coords, preds):
        pred_img[r, c] = p

    # borders
    pred_img[0,:,:] = test_raw[0,:,:]
    pred_img[-1,:,:] = test_raw[-1,:,:]
    pred_img[:,0,:] = test_raw[:,0,:]
    pred_img[:,-1,:] = test_raw[:,-1,:]

    pred_img = np.clip(pred_img, 0, 1)

    test_raw_img = (test_raw * 255).astype('uint8')
    pred_img = (pred_img * 255).astype('uint8')
    test_ref_img = (test_ref * 255).astype('uint8')

    print("PSNR:", psnr(test_ref_img, pred_img))
    print("SSIM:", ssim(test_ref_img, pred_img, channel_axis=2))

    plt.subplot(5,3,3*i+1)
    plt.imshow(test_raw_img)
    plt.title("Raw")
    plt.axis('off')

    plt.subplot(5,3,3*i+2)
    plt.imshow(pred_img)
    plt.title("Predicted")
    plt.axis('off')

    plt.subplot(5,3,3*i+3)
    plt.imshow(test_ref_img)
    plt.title("GT")
    plt.axis('off')

plt.tight_layout()
plt.show()

# ========================================================
# 10. EVALUATION METRICS(Patch and Global statistics Model)
# ========================================================
psnr_list = []
ssim_list = []
mse_list = []
worst_psnr = []   
worst_ssim = []
best_psnr = []
best_ssim = []

for test_raw, test_ref in val_pairs:

    h, w, _ = test_raw.shape

    patches = []
    coords = []
    
    mean = test_raw.mean(axis=(0,1))
    std  = test_raw.std(axis=(0,1))

    for r in range(1, h-1):
        for c in range(1, w-1):
            patch = test_raw[r-1:r+2, c-1:c+2]
            patch_flat = patch.reshape(-1)
            patch_std_local = patch.std()
            features = np.concatenate([patch_flat, mean, std,[patch_std_local]])
            patches.append(features)
            coords.append((r, c))

    patches = np.array(patches)
    pred_img = np.zeros((h, w, 3))
    pred_img[0:1, :, :] = test_raw[0:1, :, :]   
    pred_img[-1:, :, :] = test_raw[-1:, :, :]   
    pred_img[:, 0:1, :] = test_raw[:, 0:1, :]   
    pred_img[:, -1:, :] = test_raw[:, -1:, :]
    preds = model1.predict(patches)

    for (r, c), p in zip(coords, preds):
        pred_img[r, c] = p

    pred_img = np.clip(pred_img, 0, 1)

    pred_img_uint8 = (pred_img * 255).astype('uint8')
    test_ref_uint8 = (test_ref * 255).astype('uint8')

    pred_img_uint8 = cv2.GaussianBlur(pred_img_uint8,(3,3),0.5)
    pred_img_blur = pred_img_uint8.astype(np.float32) / 255.0

    p1 = psnr(test_ref_uint8, pred_img_uint8)
    s = ssim(test_ref_uint8, pred_img_uint8, channel_axis=2)

    psnr_list.append(p1)
    ssim_list.append(s)

    mse = np.mean((pred_img_blur - test_ref) ** 2)
    mse_list.append(mse)

    sample = {
        "psnr": p1,
        "ssim": s,
        "raw": test_raw,
        "pred": pred_img_blur,
        "ref": test_ref
    }
   
    best_psnr.append(sample)
    best_psnr = sorted(best_psnr, key=lambda x: -x["psnr"])[:3]

    best_ssim.append(sample)
    best_ssim = sorted(best_ssim, key=lambda x: -x["ssim"])[:3]
    worst_psnr.append(sample)
    worst_psnr = sorted(worst_psnr, key=lambda x: x["psnr"])[:3]

    worst_ssim.append(sample)
    worst_ssim = sorted(worst_ssim, key=lambda x: x["ssim"])[:3]
avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)


print("Average MSE:", np.mean(mse_list))
print("Min MSE:", np.min(mse_list))
print("Max MSE:", np.max(mse_list))

print("Average PSNR:", avg_psnr)
print("Average SSIM:", avg_ssim)

print("Min PSNR:", np.min(psnr_list))
print("Max PSNR:", np.max(psnr_list))

print("Min SSIM:", np.min(ssim_list))
print("Max SSIM:", np.max(ssim_list))

# =====================================================
# 11. VISUALIZATION(Best 3 and Worst 3 Generated Images)
# =====================================================
def plot_samples(samples, title):
    n = len(samples)
    plt.figure(figsize=(12, 4*n))

    for i, item in enumerate(samples):

        raw = (item["raw"] * 255).astype('uint8')
        pred = (item["pred"] * 255).astype('uint8')
        ref = (item["ref"] * 255).astype('uint8')

        plt.subplot(n, 3, 3*i + 1)
        plt.imshow(raw)
        plt.title("Raw")
        plt.axis('off')

        plt.subplot(n, 3, 3*i + 2)
        plt.imshow(pred)
        plt.title(f"Pred\nPSNR={item['psnr']:.2f}, SSIM={item['ssim']:.3f}")
        plt.axis('off')

        plt.subplot(n, 3, 3*i + 3)
        plt.imshow(ref)
        plt.title("Ground Truth")
        plt.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
plot_samples(best_psnr, "Top 3 Best Performing Images (PSNR)")
plot_samples(worst_psnr, "Top 3 Worst Performing Images (PSNR)")    

   




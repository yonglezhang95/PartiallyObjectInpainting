# Recovering-Partially-Corrupted-Objects-via-Sketch-Guided-Bidirectional-Feature-Interaction

---

# Data Preparation for Partially Occluded Object Inpainting

This repository includes the data preparation pipeline for constructing **partially occluded object masks** and **partial sketches**. The process involves the following steps:

## 1. Data Preparation Steps

1. **Mask Generation**
2. **Partial Masking**
3. **Partial Sketch Generation**

We provide scripts to generate test sets for both **CUB-Sketch** and **MSCOCO-Sketch**, each equipped with single-type sketches.

### Scripts Location

* [`data_preparation_process/generate_test_mask_pairs_of_cubbirds.py`](data_preparation_process/generate_test_mask_pairs_of_cubbirds.py)
* [`data_preparation_process/generate_test_mask_pairs_of_mscocoval.py`](data_preparation_process/generate_test_mask_pairs_of_mscocoval.py)

---

## 2. Download Test Data

You can download the required test datasets from our [Google Drive folder](https://drive.google.com/drive/folders/1GyooeQyxYu_LEQgbSH9go2-Ln7Vg2lE5?usp=sharing):

* `CUB-Sketch/test/CUB-Sketch-single/test_data.tar`
* `MSCOCO-Sketch/test/MSCOCO-Sketch-single/filtered_coco_val2014_with_edges_underTwoAnns.tar`

---

## 3. Generate Test Sets

### 3.1 CUB-Sketch Test Set

To generate the CUB-Sketch test set, run:

```bash
python generate_test_mask_pairs_of_cubbirds.py \
    --train_data_dir "CUB-Sketch/test/CUB-Sketch-single/test_data.tar" \
    --save_dir "your_save_dir"
```

* You can use the default values for `--area_range`, `--dilate_interval_len`, and `--gaussian_blur_interval_len`, or specify custom parameters.

### 3.2 MSCOCO-Sketch Test Set

To generate the MSCOCO-Sketch test set, run:

```bash
python generate_test_mask_pairs_of_mscocoval.py \
    --train_data_dir "MSCOCO-Sketch/test/MSCOCO-Sketch-single/filtered_coco_val2014_with_edges_underTwoAnns.tar" \
    --save_dir "your_save_dir"
```

* Default values for the optional parameters (`--area_range`, `--dilate_interval_len`, `--gaussian_blur_interval_len`) are provided, but you may override them as needed.

---

## 4. Coming Soon

We will release the **complete training code** and **training dataset** for partially corrupted object inpainting after the paper is accepted.

---


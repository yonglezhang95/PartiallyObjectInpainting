# Recovering-Partially-Corrupted-Objects-via-Sketch-Guided-Bidirectional-Feature-Interaction

---

1. **Data Preparation for Constructing Partially Occluded Object Masks and Partial Sketches**

   The data preparation process involves the following three steps:

   1. **Mask Generation**
   2. **Partial Masking**
   3. **Partial Sketch Generation**

   We provide code to generate the CUB-Sketch and MSCOCO-Sketch test sets, each containing a single-type sketch. These scripts can be found in the following directories:

   * `data_preparation_process/generate_test_mask_pairs_of_cubbirds.py`
   * `data_preparation_process/generate_test_mask_pairs_of_mscocoval.py`

   **1.2** You can download the required test datasets from the following link:
   [Download CUB-Sketch and MSCOCO-Sketch test sets](https://drive.google.com/drive/folders/1GyooeQyxYu_LEQgbSH9go2-Ln7Vg2lE5?usp=sharing)

   * `CUB-Sketch/test/CUB-Sketch-single/test_data.tar`
   * `MSCOCO-Sketch/test/MSCOCO-Sketch-single/filtered_coco_val2014_with_edges_underTwoAnns.tar`

   **1.3** To generate the CUB-Sketch test set, run the following command:

   ```bash
   python generate_test_mask_pairs_of_cubbirds.py \
       --train_data_dir "CUB-Sketch/test/CUB-Sketch-single/test_data.tar" \
       --save_dir "your_save_dir"
   ```

   *You may use the default parameters for `--area_range`, `--dilate_interval_len`, and `--gaussian_blur_interval_len`, or specify your own.*

   **1.4** To generate the MSCOCO-Sketch test set, run:

   ```bash
   python generate_test_mask_pairs_of_mscocoval.py \
       --train_data_dir "MSCOCO-Sketch/test/MSCOCO-Sketch-single/filtered_coco_val2014_with_edges_underTwoAnns.tar" \
       --save_dir "your_save_dir"
   ```

   *Default values for `--area_range`, `--dilate_interval_len`, and `--gaussian_blur_interval_len` are available, but custom values can also be specified.*

2. **Note:** The complete training code and training dataset for partially corrupted object inpainting will be released after the paper is accepted.

---

Let me know if you need a version formatted for README.md or publication.

    

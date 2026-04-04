# MultiWeather-ThermalUAV

A scene-consistent multi-weather multi-modal dataset for UAV semantic segmentation.

## Overview

MultiWeather-ThermalUAV extends the [Safe-UAV](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Marcu_SafeUAV_Learning_to_estimate_depth_and_safe_landing_areas_for_ECCVW_2018_paper.pdf) 
dataset with synthetically generated thermal imagery and four weather conditions: 
clear, fog, rain, and snow. Its defining property is **scene consistency**: geometry, 
object placement, and segmentation annotations are identical across all weather 
conditions for every scene, enabling controlled isolation of weather-induced domain 
shift from scene-level variation.

## Dataset

The dataset is hosted on Hugging Face:
[https://huggingface.co/datasets/shashankborse/multiweather-thermaluav](https://huggingface.co/datasets/shashankborse/multiweather-thermaluav)

### Statistics

| Split      | Scenes | Weather Conditions | Modalities | Total Images |
|------------|--------|--------------------|------------|--------------|
| Train      | 9,524  | 4                  | 3          | 114,288      |
| Validation | 1,190  | 4                  | 3          | 14,280       |
| Test       | 1,193  | 4                  | 3          | 14,316       |
| **Total**  | **11,907** | **4**          | **3**      | **142,884**  |

### Modalities
- **RGB**: Visible light imagery
- **Thermal**: Synthetically generated 16-bit thermal imagery
- **Masks**: Semantic segmentation masks with 3 classes

### Classes
| Class | Label | Description |
|-------|-------|-------------|
| 0 | Horizontal | Safe landing zones |
| 1 | Vertical | Obstacles |
| 2 | Other | Sloped or irregular surfaces |

### Weather Conditions
- **Clear**: Original Safe-UAV imagery
- **Fog**: Depth-based atmospheric scattering (Beer-Lambert law)
- **Rain**: Directional streak simulation with motion blur
- **Snow**: Depth-aware particle distribution

### Dataset Structure

```
final_dataset/
  {train, val, test}/
    {clear, fog, rain, snow}/
      {rgb, thermal, masks}/
        000000.png ... NNNNNN.png
```

## Code

### Requirements

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib h5py
```

### Generate Dataset from Scratch

If you have the original Safe-UAV `.h5` files, you can regenerate the dataset:

```bash
python3 generate_dataset.py
```

Place the `.h5` files (`suburbA.h5`, `suburbB.h5`, `urbA.h5`, `urbB.h5`) in the 
same directory before running.

### Assemble Final Dataset

After generation, run the assembly script to create the structured final dataset:
```bash
python3 rebuild_final_dataset.py
```

This reorganises the generated data into train, val, and test splits with 
consistent sequential naming across all weather conditions and modalities.

### Run Experiments

Open `experiments.ipynb` in Jupyter and run cells top to bottom. The notebook 
covers:

- Dataset loading and verification
- U-Net training and evaluation
- LR-ASPP MobileNetV3-Large training and evaluation
- Results table and figures

```bash
pip install jupyter
jupyter notebook experiments.ipynb
```

### Results

| Model | Modality | Clear | Fog | Rain | Snow |
|-------|----------|-------|-----|------|------|
| U-Net | RGB | 0.607 | 0.372 | 0.332 | 0.584 |
| U-Net | Thermal | 0.557 | 0.425 | 0.532 | 0.555 |
| U-Net | Fusion | **0.721** | 0.239 | 0.296 | 0.479 |
| LR-ASPP | RGB | 0.551 | 0.385 | 0.425 | 0.543 |
| LR-ASPP | Thermal | 0.524 | **0.459** | 0.515 | 0.524 |
| LR-ASPP | Fusion | 0.638 | 0.189 | 0.304 | **0.558** |

All models trained on clear weather only, evaluated across all conditions without 
adaptation.

## Key Findings

- **Thermal is the most robust modality** under fog and rain, outperforming RGB 
  significantly under adverse conditions
- **Early fusion collapses under fog**, performing worse than either single modality 
  despite achieving the highest clear weather performance
- **Snow is the mildest degradation** across all modalities and models

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{borse2026multiweather,
  title   = {MultiWeather-ThermalUAV: A Scene-Consistent Multi-Weather 
             Multi-Modal Dataset for UAV Semantic Segmentation},
  author  = {Borse, Shashank Dilip},
  journal = {Journal of Data-centric Machine Learning Research},
  year    = {2026}
}
```

## Licence

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
It is derived from the Safe-UAV dataset, made available by Marcu et al. under its 
original licence terms.

## Contact

Shashank Dilip Borse — [work@shashankborse.com](mailto:work@shashankborse.com)

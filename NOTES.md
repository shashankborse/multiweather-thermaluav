# Important Notes

## Reproducing the Dataset

The original Safe-UAV `.h5` files are required to regenerate 
the dataset from scratch. These can be obtained from the 
Safe-UAV authors. The final processed dataset is available 
directly on Hugging Face and does not require regeneration.

## Reproducing Experiments

The experiments notebook uses a fixed random seed of 42 
throughout. All results should be reproducible on any 
hardware, though absolute training time will vary.

MPS (Apple Silicon) was used for the reported results. 
The code automatically detects and uses CUDA if available, 
falling back to MPS then CPU.

## Dataset Access

The final_dataset folder structure expected by experiments.ipynb:

final_dataset/
  {train, val, test}/
    {clear, fog, rain, snow}/
      {rgb, thermal, masks}/
        000000.png ... NNNNNN.png

Download from Hugging Face:
https://huggingface.co/datasets/shashankborse/multiweather-thermaluav

## Known Issues

RGB images are stored in BGR byte order as saved by OpenCV 
and loaded accordingly in the experiment pipeline using 
cv2.COLOR_BGR2RGB. This is handled automatically in the 
dataset class.

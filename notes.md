Datasets:

In the Physionet 2020 Challenge there are 5+2 datasets (https://physionet.org/content/challenge-2020/1.0.2/). Each dataset
can have any of 112 labels found here (https://github.com/physionetchallenges/physionetchallenges.github.io/blob/master/2020/Dx_map.csv).
Every dataset is using 12-lead ECG data with typically one but sometimes multiple classifications per record. 
- CPSC Database and CPSC-Extra Database
  - CPSC has 6877 Samples
  - CPSC-Extra has 3453 Samples
  - 500 Hz
  - 6-60 Seconds in length
- INCART Database
  - 75 Samples
  - 30 Minutes Each
  - 257 Hz
  - 175,000 total beats
- PTB and PTB-XL Database:
  - PTB has 516 Records sampled at 1000 Hz (doesn't state how long)
  - PTB-XL has 21,837 samples sampled at 500 Hz for 10 Seconds
- Georgia ECG Challenge Database
  - 10,344 Samples
  - 500 Hz
  - 10 Seconds long


Other:
- mitbih Dataset
  - 48 Half-hour exceprts of two-channel ambulatory ECG
  - 360 hz
  - 2 Lead
  - Each beat annotated for a total of 110,000 total beats


![classes](https://github.com/VertAdam/Arrhythmia-Detection-Transformer/blob/main/plots/report_figs/classes.png)
![architecture](https://github.com/VertAdam/Arrhythmia-Detection-Transformer/blob/main/plots/report_figs/architecture.png)
![recall](https://github.com/VertAdam/Arrhythmia-Detection-Transformer/blob/main/plots/report_figs/recall.png)
![precision](https://github.com/VertAdam/Arrhythmia-Detection-Transformer/blob/main/plots/report_figs/precision.png)
# Towards Urban Fingerprinting of NYC 
This codebase is in support of FARLAB's XCPedestrian project, where we seek to quantify differences in pedestrian-vehicle interactions across different locales. 

## Contents 
### Scripts 
This contains code for CV processing -- pipeline.py is the main script we use to (1) run COCO-trained YOLO on urban footage, returning instance counts of 80 classes; including pedestrians, cars, and bikes; and (2) isolate relevant clips that contain intersections for downstream qualitative coding. 

### Jobs 
This contains SLURM job scripts for triggering the scripts in an HPC environment. Modify these as needed for your own system. 

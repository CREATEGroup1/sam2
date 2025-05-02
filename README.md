First set up conda environment:

```
conda env create -f environment.yaml
```

To visualize ultrasound segmentations for all frames of a video:

```
python eval_one_frame.py /Users/emma/Desktop/QUEENS/CREATE_CHALLENGE/Training_Data/AN02-20210107-131328/AN02-20210107-131328_0041.jpg 
```


To generate and save masks for a single video as npy file:

```
python auto_segment_video_masks.py <path_to_video_folder> 
```

To visually inspect a npy file of video masks:
```
python view_segmentations_npy.py /path/to/video_folder /path/to/Group_1_Subtask3_<VideoId>.npy
```

To combine all video npy files into one for submission:
```
python combine_segmentations.py /path/to/folder/of/npy/files
```
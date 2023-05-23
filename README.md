## Project name: 2023_asd_socialgaze
* author: Chanyoung Ko
* date: 2023-05-23
* python version: 3.9.12
* opencv version: 4.7.0
* open3d version: 0.17.0
* azure kinect sdk version: 1.4.1


## Objective
- Create social gaze-based subclassification of Autism Spectrum Disorder(ASD), utilizing RGB-D data acquired during joint attention situations.

## Project structure
* code
    * src 
    * data
        * raw_data
            1. ija
            2. rja_high
            3. rja_low 
        * proc_data
            1. proc_ija
            2. proc_rja_high
            3. proc_rja_low
    

## Dataset
### [`dataset/participant_information_df.csv`](dataset/participant_information_df.csv)
      
## Labels
* Binary: 0 - good responder, 1 - poor responder
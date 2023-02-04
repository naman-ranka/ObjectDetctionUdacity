# Object Detection in an Urban Environment

![This is an image](/home/animation.gif)

### Project overview
This section should contain a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self driving car systems?

In this project, we create an object detection model that can classify and detect - cars, pedestrians and cyclists from the camera input. 
First step of the poject is to understand the data we are working with. We do this in the [exploratory data analysis](/Exploratory%20Data%20Analysis.ipynb) notebook. We explore the given dataset by ploting random images. The distribution of different classes and also distribution of day versus night images have been plotted. 
The second step is to create a object detection model. This is achieved by a machine learning method called transfer learning, where we choose a pre trained model from Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and train it on their own data set. I have trained my model three times for around a collective of 50,000 steps. The results of the model's performance are discussed below.



### Set up
This section should contain a brief description of the steps to follow to run the code for this repository.

**The poject was caried on my local setup (Windows Machine)**

**Follow these steps to download and work with my trained model**

- Clone the project
```
git clone https://github.com/naman-ranka/nd013-c1-vision-starter-solution.git
```

- Install the dependencies from requirments.txt
```
pip install requirements.txt
```

- Download the dataset from [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)

- Create folders for Data Structure
```
/home/data/
    - train: contain the train data 
    - val: contain the val data 
    - test - contains 3 files to test your model and create inference videos
```
- Download the pretrained model from [here](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to /home/experiments/pretrained_model/:

- Edit config file
```
python edit_config.py --train_dir /home/data/train/ --eval_dir /home/data/val/ --batch_size 2 --checkpoint /home/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/experiments/label_map.pbtxt
```
Here you can change --checkpoint path based on the checkpoint you want to start training your model from.

- Training:

command for training:

```
python experiments/model_main_tf2.py --model_dir=experiments/try1/ --pipeline_config_path=experiments/try1/pipeline_new.config
```

- Once the training is finished, launch the evaluation process:

Ex command for evaluation:
```
python experiments/model_main_tf2.py --model_dir=experiments/try1/ --pipeline_config_path=experiments/ty1/pipeline_new.config --checkpoint_dir=try1/reference/
```

- To monitor the training luanch tensorboard instance:
```
python -m tensorboard.main --logdir experiments/try1/
```

- Export the trained model:
```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/try1/pipeline_new.config --trained_checkpoint_dir experiments/try1/ --output_directory experiments/try1/exported/
```

- Creating Animation:
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/try1/exported/saved_model --tf_record_path data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/try1/pipeline_new.config --output_path animation.gif
```

### Dataset
#### Dataset analysis
Dataset analysis is done in [Exporatory Data Analysis.ipynb](/Exploratory%20Data%20Analysis.ipynb) notebook.

- **20 random images are displayed.**

This is done to understand what kind of data we are working with.
Analysis - Images present in the dataset are of size-**640,640,3** (RGB).

![This is an image](/home/images/ExploratoryDataAnalysis.png)


- **Distribution of classes in the given dataset.**

Analysis - Images mostly contain cars.Pedestians are present in about 20% of the given dataset. No of cyclist in the given dataset is extremely low and traing a model to detect cyclist from this dataset will be challenging.

![This is an image](/home/images/ExploratoryDataAnalysis1.png)


- **Disribution of Day vs Night images.**

Analysis - 90% of images are bright (day) and 10% are dim (night) 

![This is an image](/home/images/ExploratoryDataAnalysis2.png)






### Training

#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

Pipeline file - [/home/pipeline_files/pipline_try1.config](/home/pipeline_files/pipeline_try1.config)

- ***Optimizer*** - momentum optimizer
- ***Initial learing rate*** - 0.13
- ***No of steps*** - 5000

```
data_augmentation_options {
    random_horizontal_flip {
    	probability:0.01
    }
}
data_augmentation_options {
	random_adjust_brightness{
    	max_delta :0.2
    }	
}
data_augmentation_options {
    random_crop_image {
      min_object_covered: 4
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.8
      max_area: 1.0
      overlap_thresh: 0.3
    }
}    
```
#### Results - [Tensorboard](https://tensorboard.dev/experiment/nWNlC3ecQhyBMcvmf3wspQ/)

- **Loss**

![This is an image](/home/images/try1_loss.png)

- **Precision**

![This is an image](/home/images/try1_precision.png)

- **Recall**

![This is an image](/home/images/try1_recall.png)

- **animation**

![Animation](/home/experiments/try1/animation_try1_segment-11918003324473417938_1400_000_1420_000_with_camera_labels.gif)




### Experiment 2

Pipeline file - [/home/pipeline_files/pipline_try2.config](/home/pipeline_files/pipeline_try2.config)

- ***Optimizer*** - adam optimizer
- ***Initial learing rate*** - 0.0002  
- ***No of steps*** - 5000

```
  data_augmentation_options {
	random_adjust_brightness{
    	max_delta :0.1
    }	
  }
```
#### Results - [Tensorboard](https://tensorboard.dev/experiment/nsWOiNa8Qo6EqeON6qOi8w/)

- **Loss**

![This is an image](/home/images/try2_loss.png)

- **Precision**

![This is an image](/home/images/try2_precision.png)

- **Recall**

![This is an image](/home/images/try2_recall.png)

- **animation**

![Animation](/home/experiments/try2/animation_try2_segment-11918003324473417938_1400_000_1420_000_with_camera_labels.gif)


### Experiment 3

Pipeline file - [/home/pipeline_files/pipline_try3.config](/home/pipeline_files/pipeline_try3.config)

- ***Optimizer*** - adam optimizer
- ***Initial learing rate*** - 0.0002 
- ***No of steps*** - 5000

```
  data_augmentation_options {
	random_adjust_brightness{
    	max_delta :0.1
    }	
  }
 
 data_augmentation_options {
    random_crop_image {
      min_object_covered: 4
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.8
      max_area: 1.0
      overlap_thresh: 0.3
    }
  }    

data_augmentation_options {
    random_adjust_contrast{
    	min_delta: 0.8
        max_delta: 1.25
    }
  }
data_augmentation_options {
    random_adjust_saturation{
    	min_delta: 0.8
        max_delta: 1.25
    }
  }
```

#### Results - [Tensorboard](https://tensorboard.dev/experiment/alXdD06jTQWPIREp9YqO9w/)

- **Loss**

![This is an image](/home/images/try3_loss.png)

- **Precision**

![This is an image](/home/images/try3_precision.png)

- **Recall**

![This is an image](/home/images/try3_recall.png)

- **animation**

![Animation](/home/experiments/try3/animation_try3_segment-11918003324473417938_1400_000_1420_000_with_camera_labels.gif)



### Experiment 4

Pipeline file - [/home/pipeline_files/pipline_try4.config](/home/pipeline_files/pipeline_try4.config)

- ***Optimizer*** - adam optimizer
- ***Initial learing rate*** - 0.0001 
- ***No of steps*** - 40000

```
  data_augmentation_options {
    random_horizontal_flip {
    	probability:0.01
    }
  }    
  data_augmentation_options {
	random_adjust_brightness{
    	max_delta :0.1
    }	
  data_augmentation_options {
    random_adjust_contrast{
    	min_delta: 0.1
        max_delta: 0.3
    }
  }
  data_augmentation_options {
    random_adjust_saturation{
    	min_delta: 0.1
        max_delta: 0.3
    }
  }
```

#### Results - [Tensorboard](https://tensorboard.dev/experiment/EP56OknySQeA8QHJGeEMBw/)

- **Loss**

![This is an image](/home/images/try4_loss.png)

- **Precision**

![This is an image](/home/images/try4_precision.png)

- **Recall**

![This is an image](/home/images/try4_recall.png)

- **animation**

![Animation](/home/experiments/try4/animation_try4_segment-11918003324473417938_1400_000_1420_000_with_camera_labels.gif)


#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

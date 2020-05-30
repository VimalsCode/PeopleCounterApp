# People Counter Application

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Model selection
The people counter application is developed based on public object detection model provided by Tensorflow detection model zoo. The following was considered for the mentioned people counter use case,
- ssd_mobilenet_v2_coco : http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

- faster_rcnn_inception_v2_coco :  http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

ssd_mobilenet_v2_coco model had some issues with model accuracy as it was not able to detect all the people from the input video stream. This experiment was followed by using the faster_rcnn_inception_v2_coco model which resulted in better accuracy and was able identify all the people from the input video stream.

The faster_rcnn_inception_v2_coco can be downloaded using the following command,
```
wget  http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

The downloaded model in tar.gz format is extracted using the following command,
```
tar -xvf faster_rcnn_inception_v2_coco_2018_01_28
```

This resulted in extraction of the necessary model files,
- frozen_inference_graph.pb : frozen graph proto with weights
- pipeline.config : config file which was used to generate the graph

## Custom Layers
The development did not address any Custome Layers.
The model was converted into Intermediate Representation format using the following command,

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --output_dir faster_rcnn_inception_optimized --tensorflow_object_detection_api_pipeline_config faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
```
This generates the pair of files describing the model(.xml & .bin) which can be read, loaded, and inferred with the Inference Engine.

The Intermediate Engine core object is created along with the Intermediate Engine network object is created based on Intermediate representation artifacts.Once created, based on the mentioned device type the network Intermediate representation is queried for unsupported layers a mechanism to understand what network layers are supported in the current configuration.Additional device extension is added to address unsupported layers.

## Comparing Model Performance

After handling framework-agnostic Intermediate Representation (IR) format, the Inference Engine consumes the IR to perform inference.The initial model performance deals with the batching which impacts computational aspect during inference.The model inference was performed with single input.

The size of the model pre- and post-conversion is mentioned here. For pre-conversion frozen_inference_graph.pb is considered and for post-conversion frozen_inference_graph.xml is considered. It is visible to note that the usage of OpenVino IR yields compressed size.
Model | Type | size
------------ | ------------- | -------------
ssd_mobilenet_v2_coco | original | 67 MB
ssd_mobilenet_v2_coco | converted | 100 KB
faster_rcnn_inception_v2_coco | original | 55MB
faster_rcnn_inception_v2_coco | converted | 123KB

During the inference test with ssd_mobilenet_v2_coco and faster_rcnn_inception_v2_coco the main problem arised with the detection of the second person in the video. The corresponding frame is between 227-449 based on the experiment conducted as mentioned in original_detection_model_inference.ipynb. The accuracy calculation was performed only taking into consideration the 2nd person. This gives us the overview about the accuracy improvement with OpenVino toolkit.

Model | Type | Accuracy (2nd Person detection) | Total Inference Time (ms) | Test Environment
------------ | ------------- | ------------- | ------------- | -------------
ssd_mobilenet_v2_coco | original | True positive = 35 / 222 = 0.1576 <br> False negative = 188 / 222 = 0.8468 |107585.649 | local development environment Intel i7 / 16GB
ssd_mobilenet_v2_coco | converted | True positive = 198 / 222 = 0.8919 <br> False Negative = 24 / 222 = 0.1081 |100134.695 | udacity workspace
faster_rcnn_inception_v2_coco | original | True positive = 198 / 222 = 0.8919 <br> False Negative = 24 / 222 = 0.1081 |804680.068 | local development environment Intel i7 / 16GB
faster_rcnn_inception_v2_coco | converted | True positive = 213 / 222 = 0.9594 <br> False Negative = 9 / 222 = 0.0405 | 1276130.464 | udacity workspace


The inference time of the model pre- and post-conversion clearly shows that faster_rcnn_inception_v2_coco requires additional inference time due to it's large number of model parameter which yields better accuracy.

## Assess Model Use Cases

The people counter app can play an important role in smart city based applications. For example, particular area within a city where people density is analyzed over period of time or how public transporation is utilized by the people and where additional transportation is required. It will help definitely with better city planning.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

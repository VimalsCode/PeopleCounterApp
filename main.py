"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # variable
    last_count = 0
    total_count = 0
    start_time = 0
    image_mode = False
    
    # Set Probability threshold for detections
    global width, height, prob_threshold
    prob_threshold = args.prob_threshold
    
    # Initialise the class
    infer_network = Network()

    ### TODO: Load the model through `infer_network` ###
    model = args.model
    infer_network.load_model(model, args.device, args.cpu_extension) 

    ### TODO: Handle the input stream ###
    net_input_shape = infer_network.get_input_shape()
    #print(net_input_shape)
    in_shape = net_input_shape['image_tensor']
    
    # check if the input is a web cam
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        image_mode = True
    # Checks for video file
    else:
        # get the input value
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    
       
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)
    
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    # Loop until stream is over
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        image = cv2.resize(frame, (in_shape[3], in_shape[2]))
        # Change format from HWC to CHW
        image_to_infer = image.transpose((2, 0, 1))
        image_to_infer = image_to_infer.reshape(1, *image_to_infer.shape)
        
        request_id=0
        # Start asynchronous inference for specified request
        net_input = {'image_tensor': image_to_infer ,'image_info': image_to_infer.shape[1:]}
        #print(net_input)
        inf_start = time.time()
        duration_report = None
        infer_network.exec_net(request_id, net_input)
        # Wait for the result 
        if infer_network.wait(request_id) == 0:
            # Results of the output layer of the network
            det_time = time.time() - inf_start
            result = infer_network.get_output()
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            #print(inf_time_message)
            
            #frame, current_count = generate_out(frame, result)
            frame, current_count = generate_rcnn_out(frame, result)                    
           
            if key_pressed == 27:
                break
        
        # Send frame to the ffmpeg server
        #  Resize the frame
        frame = cv2.resize(frame, (768, 432))
        #print(frame.shape)
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
    
    cap.release()
    cv2.destroyAllWindows()

    ### TODO: Loop until stream is over ###

        ### TODO: Read from the video capture ###

        ### TODO: Pre-process the image as needed ###

        ### TODO: Start asynchronous inference for specified request ###

        ### TODO: Wait for the result ###

            ### TODO: Get the results of the inference request ###

            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###

def publish_topic(topic_name, value):
    """
    To publish the topic to MQTT server
    :return: None
    """
    client.publish(topic_name, payload=json.dumps(value), qos=0, retain=False)

def generate_ssd_out(frame, result):
    """
    Parse inference output.
    :param frame: input frame from camera/video
    :param result: list contains the data to parse output
    :return: frame
    """
    current_count = 0
    for obj in result[0][0]:
        if obj[2] >= prob_threshold:
            xmin = int(obj[3] * width)
            ymin = int(obj[4] * height)
            xmax = int(obj[5] * width)
            ymax = int(obj[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
    return frame, current_count

def generate_rcnn_out(frame, result):
    current_count = 0
    probs = result[0, 0, :, 2]
    for i, p in enumerate(probs):
        if p > prob_threshold:            
            box = result[0, 0, i, 3:]
            loc1 = (int(box[0] * width), int(box[1] * height))
            loc2 = (int(box[2] * width), int(box[3] * height))
            frame = cv2.rectangle(frame, loc1, loc2, (0, 55, 255), 1)
            current_count += 1
    return frame, current_count
        
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()

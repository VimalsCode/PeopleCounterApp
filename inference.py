#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU", CPU_EXTENSION=None):
        """
        To load the specified model and check for supported layers.
        
        :param model: path where model available in IR format
        :param device: device to be used to load the model
        :param CPU_EXTENSION: CPU extension to used
        :return: None
        """
        # Initialize the plugin
        self.plugin = IECore()
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)                             
        # Check for supported layers
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
          
        if len(unsupported_layers) != 0:
             # Add necessary extensions
            self.plugin.add_extension(CPU_EXTENSION, device)
            
        # Load the network into the Inference Engine
        self.exec_network = self.plugin.load_network(self.network, device)
                
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        # sucessfully loaded inference plugin       
        return

    def get_input_shape(self):
        """
        Provides the shape of input to the network.
        
        :return: Return the shape of the input layer
        """
        #return self.network.inputs[self.input_blob].shape
        input_shapes = {}
        for input in self.network.inputs:
            input_shapes[input] = (self.network.inputs[input].shape)
        return input_shapes

    def exec_net(self, request_id, net_input):
        """
        Perform the inference request.
        
        :param requestId: inference requested ID
        :param net_input: input to the model for inference
        :return: None        
        """        
        #self.infer_request = self.exec_network.start_async(request_id=requestId,inputs={self.input_blob: net_input})
        # Start an asynchronous request
        self.infer_request = self.exec_network.start_async(
                request_id, 
                inputs=net_input)
        return

    def wait(self, request_id):
        """
        Wait for the request to be complete.
        
        :param requestId: inference requested ID
        :return: status of the inference request
        """
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        #status = self.exec_network.requests[request_id].wait(-1)
        status = self.exec_network.requests[request_id].wait(-1)
        return status        

    def get_output(self):
        """
        To extract and return the output results.
        
        :return: inference output results
        """
        # Extract and return the output results
        return self.infer_request.outputs[self.output_blob]
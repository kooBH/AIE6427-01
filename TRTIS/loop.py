#!/usr/bin/env python
# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import glob,os
import librosa as rs
import argparse
import numpy as np
from numpy import dot
from numpy.linalg import norm
import sys
import gevent.ssl
import time

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

""" config.pbtxt
platform: "tensorrt_plan"
max_batch_size: 1
  input [
    {
      name: "input"
      data_type: TYPE_FP32
      dims: [ 64000 ]
    }
  ]
  output [
    {
      name: "output"
      data_type: TYPE_FP32
      dims: [ 256 ]
    }
  ]
"""
def test_infer(model_name,
               input0_data,
               model_version="",
               headers=None,
               request_compression_algorithm=None,
               response_compression_algorithm=None):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('input', [1,64000], "FP32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)

    outputs.append(httpclient.InferRequestedOutput('output', binary_data=True))
    query_params = {'test_1': 1}

    tic = time.time()
    for i in range(100) : 
      results = triton_client.infer(
          model_name,
          inputs,
          model_version=model_version,
          outputs=outputs,
          query_params=query_params,
          headers=headers,
          request_compression_algorithm=request_compression_algorithm,
          response_compression_algorithm=response_compression_algorithm)
    toc = time.time()
    print("Elapsed Time for processing {} data : {:.4f}".format(100,toc-tic))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument(
        '-H',
        dest='http_headers',
        metavar="HTTP_HEADER",
        required=False,
        action='append',
        help='HTTP headers to add to inference server requests. ' +
        'Format is -H"Header:Value".')
    parser.add_argument(
        '--request-compression-algorithm',
        type=str,
        required=False,
        default=None,
        help=
        'The compression algorithm to be used when sending request body to server. Default is None.'
    )
    parser.add_argument(
        '--response-compression-algorithm',
        type=str,
        required=False,
        default=None,
        help=
        'The compression algorithm to be used when receiving response body from server. Default is None.'
    )
    parser.add_argument(
      "--path",
      type=str,
      required=True,
      help="path of input"
    )
    parser.add_argument(
      "--enroll",
      type=str,
      required=False,
      default=None,
      help="enroll Speaker Feature"
    )
    parser.add_argument(
      "--version",
      type=str,
      default="v0"
    )

    FLAGS = parser.parse_args()
    triton_client = httpclient.InferenceServerClient(
                    url=FLAGS.url, verbose=FLAGS.verbose)

    model_name = "rawnet3"

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    if FLAGS.http_headers is not None:
        headers_dict = {
            l.split(':')[0]: l.split(':')[1] for l in FLAGS.http_headers
        }
    else:
        headers_dict = None

    
    data_in = np.random.rand(64000)
    data_in = rs.load(FLAGS.path,sr=16000,mono=True)[0]
    data_in = data_in[-64000:]
    data_in = np.expand_dims(data_in, axis=0)
    data_in = data_in.astype(np.float32)

    # Infer with requested Outputs

    results = test_infer(model_name, data_in,model_version=FLAGS.version, headers=headers_dict,
                         request_compression_algorithm=FLAGS.request_compression_algorithm,
                         response_compression_algorithm=FLAGS.response_compression_algorithm)
   
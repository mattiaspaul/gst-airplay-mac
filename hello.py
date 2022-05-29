import streamlit as st

import sys
import traceback
import argparse
import typing as typ
import time
import attr

import numpy as np
import time
import cv2
from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo
import gstreamer.utils as utils

t0 = time.time()

#beware the settings for the screen-size are device dependent
#the following are correct for iPhone 12 Mini in portrait
HEIGHT = 1080
WIDTH = 512

DEFAULT_PIPELINE = utils.to_gst_string([
    "airplaysrc",
    "queue",
    "h264parse",
    "decodebin",
    "videoconvert",
    "appsink emit-signals=True"
])

command = DEFAULT_PIPELINE#args["pipeline"]

st.title("Hello World!")
st.write("to screen share your iPhone, pull down control centre, tap the button with two interleaved reactangles and select 'gstairplay'.")
count = 0
c_option = st.selectbox('Would you like to see colors or gray?',('Colors', 'Gray'))
my_bar = st.progress(0)
img_fig = st.empty()

with GstContext():  # create GstContext (hides MainLoop)
    # create GstPipeline (hides Gst.parse_launch)
    with GstPipeline(DEFAULT_PIPELINE) as pipeline:
        appsink = pipeline.get_by_cls(GstApp.AppSink)[0]  # get AppSink
        # subscribe to <new-sample> signal
        #appsink.connect("new-sample", on_buffer, None)
        while((not pipeline.is_done)&(count<1002)):

            sample = appsink.emit("pull-sample")  # Gst.Sample
            #buffer = sample.get_buffer()  # Gst.Buffer
            if isinstance(sample, Gst.Sample):
                if(count==1):
                    t0 = time.time()
                if(count==1000):
                    t1000 = time.time()
                count += 1
                buffer = sample.get_buffer()
                buffer_size = buffer.get_size()
                array = np.ndarray(shape=(HEIGHT*3//2,WIDTH),buffer=buffer.extract_dup(0,
buffer_size),dtype=np.uint8)
                my_bar.progress(count/1002)
                if(c_option=='Colors'):
                    image_cv = cv2.cvtColor(array, cv2.COLOR_YUV2RGB_I420)
                    img_fig.image(image_cv)
                else:
                    image_cv = cv2.cvtColor(array, cv2.COLOR_YUV2GRAY_I420)
                    img_fig.image(image_cv,'gray')
                


                
                
                
st.write(count)
st.write(1002/(t1000-t0),'frames per second')


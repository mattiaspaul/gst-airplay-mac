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
import torch
import torch.nn.functional as F
import coremltools as ct

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
def simpleNorm(img):
    mean_intensity = img.mean()
    std_intensity = img.std()
    img = (img - mean_intensity) / (std_intensity + 1e-8)
    return img
t0 = time.time()
model = torch.jit.load('models/nnunet_helen_fast.pth')
mlmodel = ct.convert(model,inputs=[ct.TensorType(shape=torch.randn(1,1,256,256,).shape)],compute_units=ct.ComputeUnit.CPU_AND_GPU,minimum_deployment_target=ct.target.iOS15,convert_to="mlprogram")

st.write('mlmodel succesfully converted in ','%0.3f'%(time.time()-t0),' secs!')
st.write('you may try out the following site: https://tenor.com/view/faces-making-funny-famous-celebrities-make-face-gif-15156562')

col1, col2 = st.columns([1,4])#columns(2)
with col1:
    


    command = DEFAULT_PIPELINE#args["pipeline"]

    st.title("AirNet!")

    color = {}
    cdata = torch.zeros(3,9,1,1)

    color[0] = st.color_picker('Face Color', '#D00000')
    color[1] = st.color_picker('Left/Right Eye', '#FFE602')
    color[2] = st.color_picker('Eyebrows', '#30C200')
    color[3] = st.color_picker('Nose Color', '#6200BE')
    color[4] = st.color_picker('Upper Lip', '#006EFF')
    color[5] = st.color_picker('Inner Mouth', '#00FF7B')
    color[6] = st.color_picker('Lower Lip', '#F700FF')
    color[7] = st.color_picker('Hair Color', '#EB7110')

    for ii,i in enumerate((1,2,3,4,5,6,7,8)):
        rgb = tuple(int(color[ii].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        cdata[:,i,0,0] = torch.from_numpy(np.array(rgb).astype('float32')/255.0)
   
with col2:
    count = 0
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
                    array = np.ndarray(shape=(HEIGHT*3//2,WIDTH),buffer=buffer.extract_dup(0,buffer_size),dtype=np.uint8)
                    gray = cv2.cvtColor(array, cv2.COLOR_YUV2GRAY_I420)
                    img = torch.from_numpy(gray[256:640,64:448]).float().unsqueeze(0).unsqueeze(0)
                    img1 = F.interpolate(simpleNorm(img),size=(256,256),mode='bilinear').permute(0,1,3,2)
                    prediction = mlmodel.predict({'x_3':img1.reshape(1,1,256,256).numpy()})
                    prediction = F.interpolate(torch.from_numpy(prediction[list(prediction.keys())[4]]),size=(384,384),mode='bilinear')#.argmax(1).squeeze().t()
                    output = torch.softmax(100*prediction,1).permute(0,1,3,2)
                    mask0 = torch.zeros(1,9,384,384); mask0[:,5] = 1
                    mask0[:,5,10:-10,10:-10] = 0
                    mask1 = torch.zeros(1,9,384,384)
                    mask1[:,:5,10:-10,10:-10] = 1; mask1[:,6:,10:-10,10:-10] = 1;
                    output = output*mask1 + mask0

                    output_color = F.conv2d(output,cdata)[0].permute(1,2,0)
                    alpha = torch.clamp(.5 + 0.5*output[0,0,:,:],0.0,1.0)
                    gray1 = img.squeeze()/255
                    overlay = (gray1*alpha).unsqueeze(2) + output_color*(1.0-alpha.unsqueeze(2))

                    rgb = np.repeat(np.expand_dims(gray,2),3,axis=2)
                    rgb[256:640,64:448,:] = overlay.numpy()*255
                    rgb = rgb.astype('uint8')

                 
                    my_bar.progress(count/1002)
                    img_fig.image(rgb)
    st.write(count)
    st.write(1002/(t1000-t0))


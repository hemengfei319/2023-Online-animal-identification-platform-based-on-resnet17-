import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime
import numpy as np
import torch
from PIL import Image,ImageFont,ImageDraw#
from torchvision import transforms
import torch.nn.functional as F
from io import BytesIO
import pandas as pd
import time#
import cv2#
import os#
import mmcv#
import tempfile #
import shutil

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
font = ImageFont.truetype('simkai.ttf', 32)
idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
model = torch.load('checkpoints/best-0.895.pth', map_location=torch.device('cpu'))
model = model.eval().to(device)


def load_local_video(uploaded_video):
    bytes_data = uploaded_video.getvalue()
    print(uploaded_video.name)

    return bytes_data
def pred_single_frame(img, n=5):
    '''
    输入摄像头画面bgr-array，输出前n个图像分类预测结果的图像bgr-array
    '''
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR 转 RGB
    img_pil = Image.fromarray(img_rgb) # array 转 pil
    input_img = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
    pred_logits = model(input_img) # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算

    top_n = torch.topk(pred_softmax, n) # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze() # 解析出类别
    confs = top_n[0].cpu().detach().numpy().squeeze() # 解析出置信度

    # 在图像上写字
    draw = ImageDraw.Draw(img_pil)
    # 在图像上写字
    for i in range(len(confs)):
        pred_class = idx_to_labels[pred_ids[i]]
        text = '{:<15} {:>.3f}'.format(pred_class, confs[i])
        # 文字坐标，中文字符串，字体，rgba颜色
        draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 1))

    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR) # RGB转BGR

    return img_bgr, pred_softmax






#载入 onnx 模型，获取 ONNX Runtime 推理器
ort_session = onnxruntime.InferenceSession('Animals.onnx')
#构造输入，获取输出结果
x = torch.randn(1, 3, 256, 256).numpy()
# onnx runtime 输入
ort_inputs = {'input': x}
# onnx runtime 输出
ort_output = ort_session.run(['output'], ort_inputs)[0]
# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])


st.set_page_config(
    page_title="动物图像识别",    #页面标题       #icon
    layout="wide",                #页面布局
    initial_sidebar_state="auto"  #侧边栏
)
option = st.sidebar.title("导航栏")
sidebar = st.sidebar.radio(
    "",
    ( "首页","图片检测", "视频检测","监控源检测")
)
if sidebar == "首页":
    st.title("基于深度学习的动物图像识别方法研究")
    st.write('$\qquad$基于迁移学习实现一个:blue[ResNet18]来对九十种动物分类，只训练最后一个全连接层，冻结除最后一个全连接层外的所有层的权重，并进行微调训练。')
    st.write('&nbsp;')
    df_ceshi=pd.read_csv('各类别准确率评估指标.csv')
    st.write('目前支持的识别种类以及各类别准确率', df_ceshi,)
    st.write('&nbsp;')
if sidebar == "图片检测":
    uploaded_files = st.file_uploader("选择一张png或jpg的图片进行识别", type=['png','jpg'],accept_multiple_files=True)
    if len(uploaded_files) != 0:
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            bytes_stream = BytesIO(bytes_data)
        img = Image.open(bytes_stream)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        print(img)
        input_img = test_transform(img)
        input_tensor = input_img.unsqueeze(0).numpy()
        #ONNX Runtime预测
        # ONNX Runtime 输入
        ort_inputs = {'input': input_tensor}
        # ONNX Runtime 输出
        pred_logits = ort_session.run(['output'], ort_inputs)[0]
        pred_logits = torch.tensor(pred_logits)
        pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
        #解析预测结果
        #载入类别和对应 ID
        #idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
        y = pred_softmax.cpu().detach().numpy()[0] * 100
        x = idx_to_labels[int(np.where(y==max(y))[0])]
        st.image(img, caption=x)

if sidebar == "视频检测":
    uploaded_video = st.file_uploader(" ")
    if uploaded_video != None:
        input_video = tempfile.NamedTemporaryFile(delete=False)
        input_video.write(uploaded_video.read())
        st.video(uploaded_video.read())
        # 创建临时文件夹，存放每帧结果
        temp_out_dir = time.strftime('%Y%m%d%H%M%S')
        os.mkdir(temp_out_dir)
        print('创建临时文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))
        # 读入待预测视频
        imgs = mmcv.VideoReader(input_video.name)

        #prog_bar = mmcv.ProgressBar(len(imgs))
        i = len(imgs)
        print("i",i)
        percent_complete = 0
        progress_text = "正在处理请稍等。Operation in progress. Please wait."
        my_bar = st.progress(0)
        # 对视频逐帧处理
        for frame_id, img  in enumerate(imgs):
            ## 处理单帧画面
            img, pred_softmax = pred_single_frame(img, n=5)
            percent_complete += 100/i
            print(percent_complete)
            my_bar.progress(int(percent_complete))
            # 将处理后的该帧画面图像文件，保存至 /tmp 目录下
            cv2.imwrite(f'{temp_out_dir}/{frame_id:06d}.jpg', img)

            #prog_bar.update()  # 更新进度条

        # 把每一帧串成视频文件
        mmcv.frames2video(temp_out_dir, 'output/output_pred.mp4', fps=imgs.fps, fourcc='mp4v')
        shutil.rmtree(temp_out_dir)  # 删除存放每帧画面的临时文件夹
        print('删除临时文件夹', temp_out_dir)
        #生成视频
        video_file = open('output/output_pred.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        with open('output/output_pred.mp4') as f:
            st.download_button('Download mp3', f)








if sidebar == "监控源检测":
    img_file_buffer = st.camera_input("监控源检测")

    if img_file_buffer is not None:
        # To read image file buffer as a PIL Image:
        img_pil = Image.open(img_file_buffer)
        input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
        pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
        pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
        n = 5
        top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
        pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别
        confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析出置信度

        draw = ImageDraw.Draw(img_pil)
        # 在图像上写字
        for i in range(len(confs)):
            pred_class = idx_to_labels[pred_ids[i]]
            text = '{:<15} {:>.3f}'.format(pred_class, confs[i])
            # 文字坐标，中文字符串，字体，rgba颜色
            draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 1))
        img = np.array(img_pil)  # PIL 转 array

        # To convert PIL Image to numpy array:
        #img_array = np.array(img)

        # Check the type of img_array:
        # Should output: <class 'numpy.ndarray'>
        st.image(img)


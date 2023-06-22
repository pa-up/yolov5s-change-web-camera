import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import forms
import convert_img
import cv2
import numpy as np
import torch


st.title("カメラ映像を加工するサイト")
st.write("<p></p>", unsafe_allow_html=True)
st.write("<p></p>", unsafe_allow_html=True)

st.write("<h5>加工オプション</h5>", unsafe_allow_html=True)
# formに入力する文字列を定義
object_name_list = forms.get_form_name_list(forms.label_choice)
inside_or_background_list = forms.get_form_name_list(forms.inside_or_background)
convert_name_list = forms.get_form_name_list(forms.convert_choice)

# formを表示（検出する物体を取得と実行する画像処理の関数を取得）
choiced_object_name = st.selectbox("検出したい物体を選択", object_name_list)
inside_or_background = st.selectbox("内側と外側のどちらかを選択", inside_or_background_list)
choiced_convert_name = st.selectbox("実行したい画像加工の内容を選択", convert_name_list)

st.write("<p></p>", unsafe_allow_html=True)
display_html = \
    '<span style="font-size:75%;">検出したい物体によって、検出精度が異なります。 \
    <br>「人の顔」は基本的に画像内に締める割合が最も大きい顔のみ検出されます。 \
    <br>「人の顔」以外は画像内のあらゆる物体を検出することができます。 </span>'
st.write(display_html, unsafe_allow_html=True)


# formの情報を変数や関数として変換
choiced_object_label = forms.form_name_to_variable_data(forms.label_choice,choiced_object_name)
inside_or_background = forms.form_name_to_variable_data(forms.inside_or_background,inside_or_background)
choiced_convert_func = forms.form_name_to_variable_data(forms.convert_choice,choiced_convert_name)
# 検出物体に行う画像処理の関数と引数を決定
convert_object_func, convert_object_parameter = convert_img.back_convert_func(choiced_convert_func)

st.write("<p></p>", unsafe_allow_html=True)
st.write("<p></p>", unsafe_allow_html=True)


def convert_camera_frame(frame):
    # 画像全体に物体検出を実行する場合
    if choiced_object_label == "all":
        output_img = convert_object_func(frame, convert_object_parameter)

    # カスケード分類器で物体検出を実行する場合（choiced_object_labelは分類器へのパス）
    elif "CascadeClassifier" in choiced_object_label:
        cascade_information = np.array([
            choiced_object_label, inside_or_background, convert_object_func, convert_object_parameter,
        ])
        output_img = convert_img.cascade_convert_detect_area(frame, cascade_information)
    
    # yolov5モデルを使う場合
    else:
        # 画像処理の内容と検出物体の情報をnumpyに格納
        label_information = np.array([
            choiced_object_label, inside_or_background, convert_object_func, convert_object_parameter,
        ])
        # ONNX変換された学習済みYOLOv5モデルのファイルパス
        onnx_model_name =  'yolov5s.onnx'
        # 推論の実行
        output_img = convert_img.read_onnx_yolov5_and_detect(
            frame , onnx_model_name , label_information, 
            forms.yolov5_coco_label , confidence_threshold=0.98
        )
    
    return output_img


def callback(frame):
    input_img = frame.to_ndarray(format="bgr24")

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    output_img = convert_camera_frame(input_img) # フレームに画像処理の実行
    output_img = cv2.flip(output_img, 1)  # 画像を水平方向に反転
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    return av.VideoFrame.from_ndarray(output_img, format="bgr24")


st.write("<h5>映像の表示</h5>", unsafe_allow_html=True)
webrtc_streamer(
    key="example", video_frame_callback=callback, 
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
     media_stream_constraints={"video": True, "audio": True},
)   
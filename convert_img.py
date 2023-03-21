""" 
このファイルは静止画像に画像処理を実行する関数だけではなく、
動画・WEBカメラに対して、フレーム毎に画像処理を実行する関数も利用可
"""

import cv2
import numpy as np
from PIL import Image
import random
import pickle
import time
import forms


def no_change(cv_img, func2):
    """ 無修正 """
    return cv_img



def gray(cv_img, func2):
    """ グレースケール化 """
    cv_calc_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    cv_calc_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cv_calc_img = cv2.cvtColor(cv_calc_img, cv2.COLOR_GRAY2BGR)
    return cv_calc_img



def binar2pil(binary_img):
    """ バイナリ画像PIL画像に変換 """
    pil_img = Image.open(binary_img)
    return pil_img



def binar2opencv(binary_img):
    """ バイナリ画像をOpenCV画像に変換 """
    pil_img = Image.open(binary_img)
    cv_img = pil2opencv(pil_img)
    return cv_img



def pil2opencv(pil_img):
    """ PIL画像をOpenCV画像に変換 """
    cv_img = np.array(pil_img, dtype=np.uint8)

    if cv_img.ndim == 2:  # モノクロ
        pass
    elif cv_img.shape[2] == 3:  # カラー
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    elif cv_img.shape[2] == 4:  # 透過
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)
    return cv_img



def opencv2pil(cv_calc_img):
    """ OpenCV画像をPIL画像に変換 """
    pil_img = cv_calc_img.copy()
    
    if pil_img.ndim == 2:  # モノクロ
        pass
    elif pil_img.shape[2] == 3:  # カラー
        pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB)
    elif pil_img.shape[2] == 4:  # 透過
        pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGRA2RGBA)
    pil_img = Image.fromarray(pil_img)
    return pil_img



def max_size(cv_img , max_img_size):
    """ 
    縦横の倍率を保ちながら、画像の辺の長さの最大値を定義
    例）max_img_size = 1500 : 画像の縦または横サイズの最大値を1500に制限
    """
    rows = cv_img.shape[0]
    cols = cv_img.shape[1]
    new_row = rows
    new_col = cols
    if (rows >= cols)  and (rows > max_img_size) :
        new_row = max_img_size
        new_col = int( cols / (rows/max_img_size) )
    #
    if (cols > rows)  and (cols > max_img_size) :
        new_col = max_img_size
        new_row = int( rows / (cols/max_img_size) )
    #
    cv_img = cv2.resize( cv_img , dsize=(new_col, new_row) )
    return cv_img


def generate_input_img_path():
    # 現在時刻をシード値として使用
    random.seed(time.time())
    digits = [str(random.randint(0, 9)) for _ in range(7)]
    input_img_path = "".join(digits) + ".jpg"
    return input_img_path



def brightness(input_image , gamma):
  """ 
  画像の明るさ（輝度）を変える関数
  gamma > 1  =>  明るくなる
  gamma < 1  =>  暗くなる 
  """
  img2gamma = np.zeros((256,1),dtype=np.uint8)  # ガンマ変換初期値

  for i in range(256):
    # ガンマ補正の公式 : Y = 255(X/255)**(1/γ)
    img2gamma[i][0] = 255 * (float(i)/255) ** (1.0 /gamma)
  
  # 読込画像をガンマ変換
  gamma_img = cv2.LUT(input_image , img2gamma)
  return gamma_img


def mosaic(input_image , parameter):
  """ 
  モザイク処理（画像を縮小してモザイク効果を適用）
  parameter : リサイズにおける 縦＝横 サイズ（小さいほどモザイクが強くなる）
  例）一般的には parameter = 25 ~ 50 など
  """
  mosaic_img = cv2.resize(input_image, (parameter , parameter), interpolation=cv2.INTER_NEAREST)
  mosaic_img = cv2.resize(mosaic_img, input_image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
  return mosaic_img



def mask(input_image , threshold):
  """ 
   2値化（マスク）処理 
   threshold : しきい値（ 0 ~ 255 の 整数値）
  """
  # グレースケール化
  input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
  img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
  # 2値化
  ret, mask_img = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
  # 2値画像を3チャンネルに拡張する
  mask_img_3ch = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
  mask_img_3ch = cv2.cvtColor(mask_img_3ch, cv2.COLOR_BGR2RGB)
  return mask_img_3ch


def back_convert_func(img_conversion):
    """ 入力した文字列に応じて、任意の画像処理関数とパラメータを返す関数 """
    """ WEB上でのユーザーからのform入力に対して、コマンドを返すことにも利用可 """
    convert_img_func = no_change
    convert_object_parameter = ""
    if img_conversion == "mosaic":
        convert_img_func = mosaic
        convert_object_parameter = 25
    if img_conversion == "mask":
        convert_img_func = mask
        convert_object_parameter = 120
    if img_conversion == "light":
        convert_img_func = brightness
        convert_object_parameter = 2
    if img_conversion == "dark":
        convert_img_func = brightness
        convert_object_parameter = 0.5
    if img_conversion == "gray":
        convert_img_func = gray
        convert_object_parameter = ""
    
    return convert_img_func , convert_object_parameter


def resize_to_square(input_img , resized_length):
  """ 
  入力画像を正方形に収まるようにリサイズし、余白を黒で塗りつぶす関数
  （リサイズ後の画像を左 or 上に敷き詰め、画像を縦横の短い方向を黒で塗りつぶす）
  """
  input_height = input_img.shape[0]
  input_width = input_img.shape[1]
  
  # 入力画像が正方形の場合
  if input_width == input_height:
    resized_height , resized_width = resized_length , resized_length
    resized_input_img = cv2.resize( input_img, (resized_height , resized_width) )
    resized_square_img = resized_input_img

  # 入力画像が縦長の場合
  if  input_width < input_height:
    resized_height , resized_width = resized_length , int( input_width * resized_length / input_height )
    resized_input_img = cv2.resize( input_img, (resized_width , resized_height) )
    # 画像を正方形の左に敷き詰め、右の余白を黒で埋め尽くす
    resized_square_img = np.zeros( (resized_length , resized_length , 3) )
    resized_square_img[ : ,  : resized_width] = resized_input_img
  
  # 入力画像が横長の場合
  if  input_width > input_height:
    resized_height , resized_width = int( input_height * resized_length / input_width ) , resized_length
    resized_input_img = cv2.resize( input_img, (resized_width , resized_height) )
    # 画像を正方形の上に敷き詰め、下の余白を黒で埋め尽くす
    resized_square_img = np.zeros( (resized_length , resized_length , 3) )
    resized_square_img[ : resized_height ,  : ] = resized_input_img

  resized_square_img = resized_square_img.astype(np.float32)
  return resized_square_img , resized_input_img



def convert_detect_area(input_img , argu2):
    """
    特定の検出物体の領域に任意の画像処理を実行する関数
    argu2 = [ label_information , detect_results ]
        label_information = numpy（choiced_object_label , inside_or_background , convert_object_func , convert_object_parameter）
            inside_or_background : 文字列（"inside" or "background"）
            convert_object_func : 画像処理を実行する関数
            convert_object_parameter : 画像処理関数のパラメータ（ = 第二引数）（第一引数が入力画像numpy）
        detect_results = np.array([ boxes[: , 0] , boxes[: , 1] , boxes[: , 2] , boxes[: , 3] , classes, confidences ])
            boxes[: , 0] ~ [: , 3] : N行のint型の要素（x_min, y_min, x_max, y_max）
            classes : N行0列（ラベル名の文字列）
            confidences : N行0列（ラベルの確信度）
    """
    output_img = input_img.copy()
    label_information , detect_results = argu2[0] , argu2[1]

    choiced_object_label = label_information[0]
    inside_or_background = label_information[1]
    convert_object_func = label_information[2]
    convert_object_parameter = label_information[3]

    # 検出領域毎に場合わけ
    if detect_results.shape[1] != 0:
        for area_number in range( detect_results.shape[1] ):
            detected_label = detect_results[4][area_number]

            if detected_label == choiced_object_label:
                x_min, y_min = detect_results[0][area_number] , detect_results[1][area_number]
                x_max, y_max = detect_results[2][area_number] , detect_results[3][area_number]
                input_detection_area = input_img[y_min : y_max , x_min : x_max]
                
                # 検出領域の切り抜き画像が空であるか否か
                img_empty_valid = input_detection_area.shape[0] == 0 or input_detection_area.shape[1] == 0

                if inside_or_background == "inside" and img_empty_valid == False :
                    # 検出領域に画像加工を実施
                    converted_area = convert_object_func(input_detection_area , convert_object_parameter)
                    output_img[y_min : y_max , x_min : x_max] = converted_area
            
                if inside_or_background == "background" and img_empty_valid == False :
                    # まず全体に画像加工を行った後に、各検出領域に加工前の入力画像を適用
                    if area_number == 0:
                        output_img = convert_object_func(input_img , convert_object_parameter)
                    output_img[y_min : y_max , x_min : x_max] = input_detection_area
    return output_img



def read_onnx_yolov5_and_detect(
    input_img , onnx_model_name , label_information ,
    all_labels , confidence_threshold=0.98
    ):
    """ 
    yolov5のONNXモデルを読み込み、特定の物体にのみ物体検出・加工を実行する関数
    onnx_model_name : ONNX変換された学習済みyolov5モデルのファイルパス
    label_information = numpy配列（choiced_object_label , inside_or_background , convert_object_func , convert_object_parameter）
        choiced_object_label : 文字列（検出したい物体のラベル名）
        inside_or_background : 文字列（"inside" or "background"）
        convert_object_func : 画像処理を実行する関数
        convert_object_parameter : パラメータ（ = 画像処理関数の第二引数）（第一引数が入力画像numpy）
    all_labels : モデルの全ラベル名（N行0列のnumpy）
    """

    def yolov5s_onnx_detect(input_img , onnx_model_name):
        """ ONNXモデルを読み込み、推論を実行する関数 """
        # ONNXモデルを読み込む
        cv_dnn_torch = cv2.dnn.readNetFromONNX(onnx_model_name)
        # 入力画像の前処理
        model_input_shape = (3, 640, 640)  # モデルの入力サイズに合わせる
        resized_square_img , resized_input_img = resize_to_square(input_img , model_input_shape[1])
        preprocessing = resized_square_img.copy()
        # モデルに入力し、出力を取得する
        input_cv2_dnn = cv2.dnn.blobFromImage(preprocessing, 1/255.0, (model_input_shape[1] , model_input_shape[2]), swapRB=True, crop=False)
        cv_dnn_torch.setInput(input_cv2_dnn)
        output_cv_torch = cv_dnn_torch.forward()
        return output_cv_torch , resized_input_img

    def model_information(output_cv_torch , confidence_threshold):
        """ 
        推論結果から、検出領域の座標、クラス、確信度を取得する後処理の関数（低い確信度を間引く）
        引数 : output_cv_torch : ONNX変換されたyolov5モデルの推論結果
        戻り値（6行N列）: detect_results = np.array([ x_min , y_min , x_max , y_max , labels , confidences ])
        """
        boxes, classes, confidences = [], [], []
        for output in output_cv_torch:
            # detection: [center_x, center_y, width, height, class1_conf, class2_conf, ..., classN_conf]
            for detection in output:
                # クラスIDを取得する
                class_id = np.argmax(detection[5:])
                # 確信度を取得
                confidence = detection[5 + class_id]
                # 検出領域ボックスの座標を取得
                box = detection[:4]
                x1, y1, x2, y2 = box.astype("int")
                # 確信度の低い or 面積がゼロの物体を間引く
                if confidence >= confidence_threshold and x2 > 0 and y2 > 0:
                    classes.append(class_id)
                    confidences.append(confidence)
                    x_min , y_min , x_max , y_max = int( x1 - x2 / 2 ) , int( y1 - y2 / 2 ) , int( x1 + x2 / 2 ) ,  int( y1 + y2 / 2 )
                    boxes.append([x_min, y_min, x_max, y_max])
        boxes , labels , confidences = np.array(boxes) , np.array(classes) , np.array(confidences)
        boxes = boxes.reshape((-1, 4))  # boxesを2次元配列に変換
        # classes の全要素を ID（数値）からラベル名に変更
        labels = np.empty( (len(classes)), dtype = 'object' )
        for n in range( len(classes) ):
            labels[n] = all_labels[classes[n]][0] 
        # 検出領域の座標、クラス、確信度を一つの numpy に格納
        detect_results = np.array([ 
            boxes[: , 0] , boxes[: , 1] , boxes[: , 2] , boxes[: , 3] , 
            labels, confidences, ])
        return detect_results

    # モデルの読み込みと推論の実行
    output_cv_torch , resized_input_img = yolov5s_onnx_detect(input_img , onnx_model_name)
    # 推論結果の後処理
    detect_results = model_information(output_cv_torch , confidence_threshold)
    # 特定の検出物体領域にのみ画像処理を実行
    output_img = convert_detect_area(resized_input_img , [ label_information , detect_results ])
    return output_img



def video_information(input_video_file , output_video_file):
  """
  総再生時間、総フレーム数の表示 : frame_number , total_time
  動画の書き込み形式の取得 ： fmt , writer
  """

  # 動画をフレームに分割
  cap = cv2.VideoCapture(input_video_file)

  #動画サイズ取得
  width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
  height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )

  #フレームレート取得
  fps = cap.get(cv2.CAP_PROP_FPS)

  #フォーマット指定（動画の書き込み形式）
  fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  writer = cv2.VideoWriter( output_video_file , fmt, fps, (width, height) )

  # 表示
  print("合計フレーム数：")
  frame_number = int( cap.get(cv2.CAP_PROP_FRAME_COUNT) )
  print(f"{frame_number} 枚 \n")

  print("合計再生時間（総フレーム数 / FPS）：")
  total_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
  total_time = round(total_time , 3)
  print(f"{total_time} 秒  \n \n")

  return cap , writer , fmt , fps , width , height



def extract_frames(cap):
    """ 動画のフレーム（画像）を一つのリストに格納する関数 """
    frames_list = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames_list.append(frame)
        else:
            break
    frames_np = np.array(frames_list)
    return frames_list , frames_np


def cascade_convert_detect_area(
    input_img , 
    cascade_information , 
    ):
    """
    OpenCVのカスケード分類器を用いて、特定の物体を検出し、その領域に任意の画像処理を実行する関数
    """
    # 分類器の読み込み
    cascade = cv2.CascadeClassifier(cascade_information[0])
    # 物体検出の実行
    detected_object = cascade.detectMultiScale(input_img , scaleFactor=1.1, minNeighbors=5)

    # 検出領域の内側に画像処理を実行
    if cascade_information[1] == "inside":
        for (x, y, w, h) in detected_object:
            detection_area = input_img[y:y+h, x:x+w]
            output_img = cascade_information[2](
                detection_area , 
                cascade_information[3] ,
            )
            input_img[y:y+h, x:x+w] = output_img
    # 検出領域の外側に画像処理を実行
    if cascade_information[1] == "background":
        for (x, y, w, h) in detected_object:
            detection_area = input_img[y:y+h, x:x+w]
            input_img = cascade_information[2](
                input_img , 
                cascade_information[3] ,
            )
            input_img[y:y+h, x:x+w] = detection_area

    return input_img

# yolov5s-change-web-camera
yolov5sモデルを用いて、WEBカメラの映像を物体検出、加工するWEBアプリ

<h2>アプリの使い方</h2>
起動URL：
<br>
https://pa-up-yolov5s-change-web-camera-app-3l1d6n.streamlit.app/
<br>
<br>
WEBアプリにアクセスすると、
<ul>
<li>検出したい物体</li>
<li>検出した物体の内側と外側のどちらを加工するか</li>
<li>画像加工の内容（モザイク化、白黒化など...）</li>
</ul>
を入力できるフォームが表示されます。
入力後、映像を表示するボタンを押すと、WEBカメラが起動して上記の加工オプションで処理された映像が表示されます。


<h2>ソースコード・環境</h2>
本WEBアプリの物体の検出は、事前にローカル環境内でONNX変換したPytorchの学習済みYolov5sモデルを用いて推論を行なっています。また、アプリのフレームワークはStreamlitを用いており、Stream Cloudにデプロイしております。

# Real-ESRGANを用いた画像拡大（4倍）アプリ 
# Author : Masamune Kitajima
# Ver : 0.1a

import argparse
import cv2
import glob
import os
import io
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from flask import Flask, request, render_template, redirect
from PIL import Image
import base64

# 画像拡大処理 (Real-ESRGAN)
class esrgan():
    def __init__(self, scale):
        """Inference demo for Real-ESRGAN.
        """
        # === setting argment ===
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
        parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
        parser.add_argument('-t', '--tile', type=int, default=256, help='Tile size, 0 for no tile during testing')
        parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
        parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
        parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
        parser.add_argument(
            '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
        parser.add_argument(
            '--alpha_upsampler',
            type=str,
            default='realesrgan',
            help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
        parser.add_argument(
            '--ext',
            type=str,
            default='auto',
            help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
        
        args = parser.parse_args()

        # determine models according to model names
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        # determine model paths
        model_path = os.path.join('./weights/', "RealESRGAN_x4plus" + '.pth')
        
    # restorer
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=args.tile,
            tile_pad=args.tile_pad,
            pre_pad=args.pre_pad,
            half=args.fp32,
        )

        self.args = args
        self.upsampler = upsampler

    # action gan for webcam
    def gan_ext(self, img_cv2):
            img = img_cv2
            try:
                output, _ = self.upsampler.enhance(img, outscale=self.args.outscale)
                return output, True
            except RuntimeError as error:
                # print('Error', error)
                # print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
                return img, False

# [Web APP制御]
# FLASKインスタンス作成
app = Flask(__name__)

# アップロード可能な拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# 拡張子が適切かどうかのチェック
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Webページ上で何かアクションが発生した時の挙動
@app.route('/', methods=['GET', 'POST'])
def predicts():

    # Request: POSTの場合
    if request.method == 'POST':
        # ファイルが存在しない場合：Request元のURLにRedirect
        if 'filename' not in request.files:
            return redirect(request.url)
        
        # ファイルが存在する場合：データの取り出し
        file = request.files['filename']
        # print("File取得完了")

        # ファイル拡張子が問題ないかチェック→問題なければ以下処理実行
        if file and allowed_file(file.filename):

            # 入力画像に対する処理
            # 画像を読み込む～バッファに書き込み
            buf = io.BytesIO()
            img = Image.open(file).convert('RGB')
            img.save(buf, format='JPEG')

            # バッファからNumpy配列に変換
            img_array = np.array(bytearray(buf.getvalue()), dtype=np.uint8)
            # Numpy配列からcv2に変換
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # 推論開始
            # init
            GAN = esrgan(4)
            # gan
            output, ret = GAN.gan_ext(img_cv)

            # ret = True is successs
            if ret:
                
                # print("推論成功")
                # 推論後データを出力
                image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                # image.save("./src/result.jpg", format='JPEG')

                # 画像をバイナリデータとして取得
                buf = io.BytesIO()
                image.save(buf, format='JPEG')
                image_data = buf.getvalue()

                # 推論後画像データをbase64でエンコードしてutf-8でデコード
                base64_str = base64.b64encode(image_data).decode('utf-8')
                # HTML側のsrcの記述に合わせるために付帯情報付与する
                base64_data = 'data:image/jpeg;base64,{}'.format(base64_str)

                # result.htmlに推論後画像を渡す
                return render_template('result.html', image=base64_data)
            
        return redirect(request.url)

    # Request：GETの場合
    elif request.method == 'GET':
        print("GET index.html")
        return render_template('index.html')

# アプリケーションの実行定義
if __name__ == '__main__':
    app.run(debug=True)
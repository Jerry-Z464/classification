# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from cv2 import cv2 as cv
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask import render_template

import glob

from modeltest import jiadian
app = Flask(__name__)

path = './static/upload/'

# 路由注册
@app.route('/', methods=['GET', 'POST'])
def index(name=None):
    return render_template('index.html', name=name)


@app.route('/upload/', methods=['POST'])
def upload():
    if request.method == 'POST':
        #获取前端上传的文件
        f = request.files.getlist('file')     
        print(f)
        #保存上传的图片文件
        for files in f:
            upload_path = os.path.join(path, secure_filename(files.filename))
            files.save(upload_path)

        files = os.listdir(path)
        print(files)
        fileList = []

        for file in files:        
            upload_path = os.path.join(path, file)
            img=cv.imread(upload_path,1)
            img=np.array(img)
            label = jiadian(img)
            print(label)
            fileList.append({"label": label})
        #删除上传的图片
        fileNames=glob.glob(path + r'\*')
        for fileName in fileNames:
            os.remove(fileName)
            
        return json.dumps(fileList, ensure_ascii=False)


if __name__ == '__main__':
    app.run()

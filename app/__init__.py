from flask import Flask,render_template,redirect,request,send_from_directory,send_file

app = Flask(__name__)
UPLOAD_FOLDER = 'upload/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from .routes import *
from .model import *
from .backprop import *



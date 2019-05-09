import os
import torch
import ntpath
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import sys
pix2pixhd_dir = Path('./src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))

from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import src.config.test_opt as opt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

web_dir2 = os.path.join(opt.results_dir, opt.name, 'face')
webpage2 = html.HTML(web_dir2, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
if not os.path.isdir(os.path.join(web_dir2,'test_sync')):
    os.mkdir(os.path.join(web_dir2,'test_sync'))
if not os.path.isdir(os.path.join(web_dir2, 'test_real')):
    os.mkdir(os.path.join(web_dir2, 'test_real'))
model = create_model(opt)

for data in tqdm(dataset):
    minibatch = 1
    generated = model.inference(data['label'], data['inst'])

    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0][2:], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    visualizer.save_images(webpage, visuals, img_path)


    short_path = ntpath.basename(img_path[0])
    name = os.path.splitext(short_path)[0]
    img_path2='test_sync/'+str(name[-5:])
    visuals_face = OrderedDict([('synthesized_image', util.tensor2im(generated.data[0]))])
    visualizer.save_face_images(webpage2, visuals_face, img_path2)


webpage.save()
torch.cuda.empty_cache()

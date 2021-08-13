# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# updated, making it compatible to pytorch 1.x in a hacky way
import re
import torch
from torch.autograd import Variable as V
from torch.autograd.variable import Variable
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image
from unidecode import unidecode
import matplotlib.pyplot as plt
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnTF():
    # load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf

def load_model():
    # this model has a last conv feature map as 14x14

    model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()): 
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model
def load_img(tf, folder,label_name):
    images = []
    for foldername in os.listdir(folder):
        _label = unidecode(foldername)
        if _label == label_name:
            for filename in os.listdir(os.path.join(folder,foldername)):
                try:
                    img = pil_loader(os.path.join(folder, foldername, filename))
                    if img is not None:
                        images.append(img)
                except:
                    print('Cant import ' + filename)
    input_images = []
    for img in images:
        input_img = V(tf(img).unsqueeze(0))
        input_images.append(input_img)
    return input_images

def xulyanh(input_images):
    place = dict()
    attribute = dict()
    count = 1
    io = 0
    for input_img in input_images:
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()

        # output the IO prediction
        io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
        if io_image < 0.5:
            io = io - 1
        else:
            io = io + 1

    # output the prediction of scene category
        #print('--SCENE CATEGORIES:')
        for i in range(0, 5):
            # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
            check = 0
            for key in place.keys():
                if key == classes[idx[i]]: check = 1
            if check == 1:
                place[classes[idx[i]]] = place[classes[idx[i]]] + 1
            else:
                new = dict([(classes[idx[i]],1)])
                place.update(new)
    # output the scene attributes

        responses_attribute = W_attribute.dot(features_blobs[count])
        idx_a = np.argsort(responses_attribute)
        # print('--SCENE ATTRIBUTES:')
        #print(', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]))
        for i in range (-1,-10,-1):
            check = 0
            for key in attribute.keys():
                if key == labels_attribute[idx_a[i]]: check = 1
            if check == 1:
                attribute[labels_attribute[idx_a[i]]] = attribute[labels_attribute[idx_a[i]]] + 1
            else:
                new = dict([(labels_attribute[idx_a[i]],1)])
                attribute.update(new)
        count = count + 2
    place = sorted(place.items(), key = lambda x : x[1], reverse=True)
    attribute = sorted(attribute.items(), key = lambda x : x[1], reverse=True)
    #print(attribute)
    return io, place, attribute

def ghifile(io, place_result, attribute_result,label_name):
    f = open('File data.txt', 'a')
    f.writelines(label_name)
    f.writelines('\n\n')
    if io > 0:
        f.writelines('outdoor')
    else: f.writelines('indoor')
    f.writelines('\n\n')
    for line in place_result:
        f.writelines(line[0])
        f.writelines('\n')
    f.writelines('\n')
    for line in attribute_result:
        f.writelines(line[0])
        f.writelines('\n')
    f.writelines('\n')
    f.close()       
features_blobs = []
# load the labels
classes, labels_IO, labels_attribute, W_attribute = load_labels()

# load the model
model = load_model()

# load the transformer
tf = returnTF() # image transformer
    
# get the softmax weight
params = list(model.parameters())
weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0

def main(path,label_name):
    #load anh
    input_images = load_img(tf,path, label_name)
    io, place, attribute = xulyanh(input_images)
    #place_result = dict()
    #attribute_result = dict()
    
    place_result = []
    attribute_result = []
    for i in range(0,5):
        # dict_temp = dict([(place[i][0],place[i][1])])
        # place_result.update(dict_temp)
        place_result.append(place[i])
    #plt.bar(place_result.keys(),place_result.values())
    #plt.show()
    for i in range(0,10):
        # dict_temp = dict([(attribute[i][0],attribute[i][1])])
        # attribute_result.update(dict_temp)
        attribute_result.append(attribute[i])
    #plt.bar(attribute_result.keys(),attribute_result.values())
    #plt.show()
    
    ghifile(io, place_result,attribute_result, label_name)
    
if __name__ == "__main__":
    main()

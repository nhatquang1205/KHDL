from re import I
from numpy.core.fromnumeric import shape
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

def load_labels():
    category = 'File data.txt'
    list1 = list()
    classes = list()
    io = list()
    places = list()
    attribute = list()
    with open(category) as class_file:
        for line in class_file:
            list1.append(line.strip())
    list1 = tuple(list1)
    
    #lay ra dia diem, indoor/outdoor
    i = 0
    j = 2
    while i < len(list1):
        classes.append(list1[i])
        io.append(list1[j])
        i = i + 21
        j = j + 21
        
    #lay ra cac dac trung ve dia diem
    i = 4
    feature = []
    while i < len(list1):
        feature.append(list1[i])
        i = i + 1
        if len(feature) == 5:
            i = i + 16
            places.append(feature)
            feature = []
    i = 10
    feature = []
    while i < len(list1):
        feature.append(list1[i])
        i = i + 1
        if len(feature) == 10:
            i = i + 11
            attribute.append(feature)
            feature = []
    return classes, io, places, attribute

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def load_labels2():
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

# load the labels
classes, labels_IO, labels_attribute, W_attribute = load_labels2()

# load the model
features_blobs = []
model = load_model()

# load the transformer
tf = returnTF() # image transformer

# get the softmax weight
params = list(model.parameters())
weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0

classes1, io, places, attributes = load_labels()

def xulyanh(input_img):
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the IO prediction
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    
    # output the prediction of scene category
    _place = []
    for i in range(0, 5):
        _place.append(classes[idx[i]])
    print(_place)
    # output the scene attributes
    responses_attribute = W_attribute.dot(features_blobs[1])
    idx_a = np.argsort(responses_attribute)
    _attribute = []
    for i in range(-1,-10,-1):
        _attribute.append(labels_attribute[idx_a[i]])
    return io_image, _place, _attribute,idx[0]

def result(io_image,place,attribute):
    _x = list()
    for _place in places:
        count = 0
        for scene in _place:
            for i in range (0,5):
                if scene == place[i]: 
                    count = count + 1
                    break
        _x.append(count)
    _x = np.array(_x)
    _y = list()
    for _attribute in attributes:
        count = 0
        for var in _attribute:
            for i in range (0,9):
                if var == attribute[i]: 
                    count = count + 1
                    break
        _y.append(count)
    _y = np.array(_y)
    result = (_x+_y)/15
    for i in range(0,len(classes1)-1):
        for j in range(1,len(classes1)):
            if  float(result[i]) < float(result[j]): 
                temp = result[i]
                result[i] = result[j]
                result[j] = temp
                temp = classes1[i]
                classes1[i] = classes1[j]
                classes1[j] = temp
    for i in range (0,len(classes1)):
        print('{:.3f} -> {}'.format(result[i], classes1[i]))
    return classes1[0]
    
def main():
    img = Image.open('TestPhoto/Bãi biển Mỹ Khê42.jpg')
    input_img = V(tf(img).unsqueeze(0))
    _io_image, _place, _attribute,idx_0 = xulyanh(input_img)
    print("KẾT QUẢ DỰ ĐOÁN: ",result(_io_image,_place,_attribute))


    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx_0])
    img2 = Image.open('TestPhoto/Bãi biển Mỹ Khê42.jpg')
    img2 = np.array(img2)
    height, width = img2.shape[:2]
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result1 = heatmap * 0.4 + img2 * 0.5
    cv2.imwrite('cam.jpg', result1)
  
if __name__ == "__main__":
    main()

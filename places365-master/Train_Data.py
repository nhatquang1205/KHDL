import os
import re
import run_placesCNN_unified
from unidecode import unidecode
list = []
folder = 'photos'
for foldername in os.listdir(folder):
    _label = re.findall('[^0-9]',foldername)
    _label = ''.join(_label)
    _label = _label.split('.jpg')[0]
    _label = unidecode(_label)
    if not list.__contains__(_label):
        list.append(_label)
for label in list:
    run_placesCNN_unified.main(folder,label)
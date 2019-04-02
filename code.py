#Modules Loaded
import os
import glob
import torch
from typing import TypeVar, NamedTuple, List, Optional, Tuple
import cv2
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.autograd import Variable

#Variables Initiated
Image = np.ndarray
GradientImage = np.ndarray
Position = NamedTuple('Position', [('x', int), ('y', int)])
Stroke = NamedTuple('Stroke', [('x', int), ('y', int), ('width', float)])
Ray = List[Position]
Component = List[Position]
ImageOrValue = TypeVar('ImageOrValue', float, Image)
Gradients = NamedTuple('Gradients', [('x', GradientImage), ('y', GradientImage)])

#Preprocessing
def gamma(x, coeff=2.2):
    return x ** (1./coeff)

def gleam(im, gamma_coeff=2.2):
    im = gamma(im, gamma_coeff)
    im = np.mean(im, axis=2)
    return np.expand_dims(im, axis=2)

def open_grayscale(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = cv2.resize(im,(640,480))
    im = im.astype(np.float32) / 255.
    return gleam(im)

def get_edges(im, lo=50, hi=220, window=3):
    im = (im * 255.).astype(np.uint8)
    edges = cv2.Canny(im, lo, hi, apertureSize=window)
    return edges.astype(np.float32) / 255.

def get_gradients(im):
    grad_x = cv2.Scharr(im, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(im, cv2.CV_64F, 0, 1)
    return Gradients(x=grad_x, y=grad_y)

def get_gradient_directions(g):
    return np.arctan2(g.y, g.x)

#Stroke Width Transformation Code
def apply_swt(im,edges,gradients, dark_on_bright=True):
    swt = np.squeeze(np.ones_like(im)) * np.Infinity
    norms = np.sqrt(gradients.x ** 2 + gradients.y ** 2)
    norms[norms == 0] = 1
    inv_norms = 1. / norms
    directions = Gradients(x=gradients.x * inv_norms, y=gradients.y * inv_norms)
    rays = []
    height, width = im.shape[0:2]
    for y in range(height):
        for x in range(width):
            if edges[y, x] < .5:
                continue
            ray = swt_process_pixel(Position(x=x, y=y), edges, directions, out=swt, dark_on_bright=dark_on_bright)
            if ray:
                rays.append(ray)
    for ray in rays:
        median = np.median([swt[p.y, p.x] for p in ray])
        for p in ray:
            swt[p.y, p.x] = min(median, swt[p.y, p.x])
    swt[swt == np.Infinity] = 0
    return swt

def swt_process_pixel(pos, edges, directions, out, dark_on_bright=True):
    threshold_dir = -0.5
    height, width = edges.shape[0:2]
    gradient_direction = -1 if dark_on_bright else 1
    ray = [pos]
    dir_x = directions.x[pos.y, pos.x]
    dir_y = directions.y[pos.y, pos.x]
    assert not (np.isnan(dir_x) or np.isnan(dir_y))
    prev_pos = Position(x=-1, y=-1)
    steps_taken = 0
    while True:
        steps_taken += 1
        cur_x = int(np.floor(pos.x + gradient_direction * dir_x * steps_taken))
        cur_y = int(np.floor(pos.y + gradient_direction * dir_y * steps_taken))
        cur_pos = Position(x=cur_x, y=cur_y)
        if cur_pos == prev_pos:
            continue
        prev_pos = Position(x=cur_x, y=cur_y)
        if not ((0 <= cur_x < width) and (0 <= cur_y < height)):
            return None
        ray.append(cur_pos)
        if edges[cur_y, cur_x] < .5: 
            continue
        cur_dir_x = directions.x[cur_y, cur_x]
        cur_dir_y = directions.y[cur_y, cur_x]
        dot_product = dir_x * cur_dir_x + dir_y * cur_dir_y
        if dot_product >= threshold_dir:
            return None
        
        stroke_width = np.sqrt((cur_x - pos.x) * (cur_x - pos.x) + (cur_y - pos.y) * (cur_y - pos.y))
        for p in ray:
            out[p.y, p.x] = min(stroke_width, out[p.y, p.x])
        return ray
    assert False, 'This code cannot be reached.'
	
#Connected Components
def connected_components(swt, threshold=3.):
    height, width = swt.shape[0:2]
    labels = np.zeros_like(swt, dtype=np.uint32)
    next_label = 0
    components = [] 
    for y in range(height):
        for x in range(width):
            stroke_width = swt[y, x]
            if (stroke_width <= 0) or (labels[y, x] > 0):
                continue
            next_label += 1
            neighbor_labels = [Stroke(x=x, y=y, width=stroke_width)]
            component = []
            while len(neighbor_labels) > 0:
                neighbor = neighbor_labels.pop()
                npos, stroke_width = Position(x=neighbor.x, y=neighbor.y), neighbor.width
                if not ((0 <= npos.x < width) and (0 <= npos.y < height)):
                    continue
                n_label = labels[npos.y, npos.x]
                if n_label > 0:
                    continue
                n_stroke_width = swt[npos.y, npos.x]
                if n_stroke_width <= 0:
                    continue
                if (stroke_width/n_stroke_width >= threshold) or (n_stroke_width/stroke_width >= threshold):
                    continue
                labels[npos.y, npos.x] = next_label
                component.append(npos)
                neighbors = {Stroke(x=npos.x - 1, y=npos.y - 1, width=n_stroke_width),
                             Stroke(x=npos.x, y=npos.y - 1, width=n_stroke_width),
                             Stroke(x=npos.x + 1, y=npos.y - 1, width=n_stroke_width),
                             Stroke(x=npos.x - 1, y=npos.y, width=n_stroke_width),
                             Stroke(x=npos.x + 1, y=npos.y, width=n_stroke_width),
                             Stroke(x=npos.x - 1, y=npos.y + 1, width=n_stroke_width),
                             Stroke(x=npos.x, y=npos.y + 1, width=n_stroke_width),
                             Stroke(x=npos.x + 1, y=npos.y + 1, width=n_stroke_width)}
                neighbor_labels.extend(neighbors)
            if len(component) > 0:
                components.append(component)
    return labels, components

#Discard Bounding Boxes with no text inside
def discard_non_text(swt, labels, components):
    variance_threshold = 0.5
    img = np.zeros_like(swt, dtype=np.uint8)
    invalid_components = []  
    rect_coords = []
    for component in components:
        if len(component) > 10000:
            continue
        average_stroke = np.mean([swt[p.y, p.x] for p in component])
        variance = np.var([swt[p.y, p.x] for p in component])
        if variance < variance_threshold*average_stroke:
            invalid_components.append(component)
            continue
        points = np.array([[p.x, p.y] for p in component], dtype=np.uint32)
        x1,y1 = np.min(points,axis=0)
        x2,y2 = np.max(points,axis=0)
        if (y2-y1) >= 10 and (y2-y1) <=300:
            if (y2-y1)/(x2-x1) >=0.1 and (y2-y1)/(x2-x1) <=10:
                if (x2-x1)/(y2-y1) >=0.1 and (x2-x1)/(y2-y1) <=10:
                    if (x2-x1)**2+(y2-y1)**2>500:
                        rect_coords.append(((x1,y1),(x2,y2)))
        for p in component:
            img[p.y,p.x] = 255
    return labels, components, img, rect_coords

def bounding_box_removal(model_location,img,rect):
    model.load_state_dict(torch.load(model_location))
    bb = img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]]
    bb_gray = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    bb_new_gray = cv2.resize(bb_gray,(28,28))  
    state.append(bb_new_gray)
    state = np.expand_dims(state, axis=1) 
    test_var = Variable(torch.FloatTensor(state), requires_grad=True)
    with torch.no_grad():
        result = model(test_var)
    values, labels = torch.max(result, 1)
    return values,labels

#Combine Characters into Lines
def get_text_lines(img,rect):
    text_img = img.copy()
    height =[]
    for a in rect:
        height.append([a[0][1],a[1][1],a[0][0],a[1][0]])
    lines = {}
    i = 0
    min_height = height[i][0]
    max_height = height[i][1]
    lines[0] =[]
    for j in range(len(height)):
        if (height[j][0] >= min_height and height[j][0]<=max_height) or (height[j][1]>=min_height and height[j][1]<=max_height):
            lines[i].append(height[j])
        else:
            i+=1
            min_height = height[j][0]
            max_height = height[j][1]
            lines[i] =[]
            lines[i].append(height[j])
    for i in range(len(lines)):
        min_med_ht = np.median(np.array(lines[i]).T[0])
        max_med_ht = np.median(np.array(lines[i]).T[1])
        for j in range(len(lines[i])):
            if ((lines[i][j][1]-lines[i][j][0])> 1.5*(max_med_ht-min_med_ht)) or ((lines[i][j][1]-lines[i][j][0])< 0.75*(max_med_ht-min_med_ht)):
                lines[i][j][0] = min_med_ht
                lines[i][j][1] = max_med_ht
        min_ht = np.min(np.array(lines[i]).T[0])
        max_ht = np.max(np.array(lines[i]).T[1])
        min_wd = np.min(np.array(lines[i]).T[2])
        max_wd = np.max(np.array(lines[i]).T[3])
        if len(lines[i])>=2:
            cv2.rectangle(text_img,(int(min_wd),int(min_ht)),(int(max_wd),int(max_ht)),(255,0,0),3)
    return text_img,lines
            

#Split Lines into words
def get_words(img,lines,rect):
    width = {}
    word_img = img.copy()
    for i in range(len(lines)):
        if len(lines[i])>=2:
            width = []
            a = np.array(lines[i]).T[0];b = np.array(lines[i]).T[1];c = np.array(lines[i]).T[2];d = np.array(lines[i]).T[3]
            indices = np.argsort(c)
            line = []
            for j in indices:
                line.append(lines[i][j])
            for j in range(len(lines[i])-1):
                width.append(line[j+1][2] - line[j][3])
            min_width = np.median(width)
            min_x = line[0][2]
            min_y = np.min(np.array(lines[i]).T[0])
            max_y = np.max(np.array(lines[i]).T[1])
            for i in range(len(line)-1):
                if width[i]>3*min_width:
                    max_x = line[i][3]
                    cv2.rectangle(word_img,(int(min_x),int(min_y)),(int(max_x),int(max_y)),(255,0,0),3)
                    min_x = line[i+1][2]
            cv2.rectangle(word_img,(int(min_x),int(min_y)),(int(line[-1][3]),int(max_y)),(255,0,0),3)            
    return word_img

#Main Function 
def main(path, dark_on_bright):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(640,480))
    im = open_grayscale(path) 
    edges = get_edges(im) 
    edge_image = np.squeeze(edges)

    gradients = get_gradients(im)
    swt = apply_swt(im, edges, gradients, dark_on_bright)  

    swt_save = (255*np.squeeze(swt)/np.squeeze(swt).max()).astype(np.uint8)
    swt_image = np.squeeze(swt_save)

    ksize = 4
    swt = cv2.blur(swt,(ksize,ksize))
    labels, components = connected_components(swt) 
    labels, components, text_img, rect_coords = discard_non_text(swt, labels, components)
    labels = labels.astype(np.float32) / labels.max()

    l = (labels*255.).astype(np.uint8)
    l[l==0] = 255
    component_image = np.squeeze(swt)
    swt = (255*swt/swt.max()).astype(np.uint8)
    letters_image = cv2.imread(path)
    letters_image = cv2.resize(letters_image,(640,480))
    for coord in rect_coords:
        try: 
            values,labels = bounding_box_removal('model.pt',letters_image,rect_coords)
            if values>=0.1 or values<0.1:
                cv2.rectangle(letters_image,coord[0],coord[1],(255,0,0),3)
        except:
            cv2.rectangle(letters_image,coord[0],coord[1],(255,0,0),3)
        cv2.rectangle(letters_image,coord[0],coord[1],(255,0,0),3)
    line_image,text_lines = get_text_lines(img,rect_coords)
    word_image = get_words(img,text_lines,rect_coords)
    return img,edge_image,swt_image,component_image,letters_image,line_image,word_image

if __name__ == "__main__":
	print('Write the location of the image')	
	path = input()
	print('is the text dark on a bright background or not - Answer Yes or No')
	text_type = input()
	if text_type =='Yes':
		dark_on_bright = True
	else:
		dark_on_bright = False	
	img,edge_image,swt_image,component_image,letters_image,line_image,word_image = main(path,dark_on_bright=dark_on_bright)
	titles = ['Original Image','Edge Image','SWT Image','Letters Image','Lines Image','Words Image']
	variables = [img,edge_image,swt_image,letters_image,line_image,word_image]
	plt.figure(figsize=(32,24))
	for i in range(len(titles)):
		plt.subplot(2,3,i+1);plt.imshow(variables[i]);plt.title(titles[i],fontsize=30)
	plt.show()

	


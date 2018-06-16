import cv2
import numpy as np
import tensorflow as tf
import imgaug as ia
from imgaug import parameters as iap
from imgaug import augmenters as iaa
import os
from random import randint
import multiprocessing
import threading

class DataService(object):
    def __init__(self, source_p, do_augment, data_path, out_size, mp_params = None):
        self.source_p = source_p
        self.do_augment = do_augment
        self.data_path = data_path
        self.out_size = out_size
        self.mp = mp_params
        if self.mp is not None:
            self.q = multiprocessing.Queue(self.mp['lim'])
    
    def start(self):
        if self.mp is None: raise RuntimeError('Service was not initialised as a multi-processing obj')
        print('Starting parallelised augmentation...')
        thread = threading.Thread(target = self.spawn)
        self.is_running = True
        thread.start()      

    def stop(self):
        if self.mp is None: raise RuntimeError('Service was not initialised as a multi-processing obj')
        print('Stopping parallelised augmentation...')
        [p.terminate() for p in self.proc_lst]
        self.is_running = False

    def pop(self):
        if self.mp is None: raise RuntimeError('Service was not initialised as a multi-processing obj')
        while True:
            if not self.q.empty():
                return self.q.get()

    def spawn(self):
        if self.mp is None: raise RuntimeError('Service was not initialised as a multi-processing obj')
        print('Spawning', self.mp['n'], 'workers')
        self.proc_lst = []
        for i in range(self.mp['n']):
            p = multiprocessing.Process(target = self.worker)
            self.proc_lst.append(p)
            p.start()
        while self.is_running:
            pass
        [p.join() for p in self.proc_lst]
        self.proc_lst = []

    def worker(self):
        np.random.seed()
        while True:
            imgs, boxes = self.random_sample(self.mp['b_s'], False)
            if not self.q.full():
                self.q.put(tuple([imgs, boxes]))
            
    def read_image(self, loc):
        img = cv2.imread(self.data_path + loc.strip())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def random_sample(self, count, ret_orig = False):
        choices = np.random.choice(self.source_p, size = count)
        imgs = [self.read_image(i['file_path']) for i in choices]
        boxes = [i['bbox'] for i in choices]
        imgs, boxes = self.resize_images(imgs, boxes)
        imgs_orig = imgs[:]
        boxes_orig = boxes[:]
        if self.do_augment:
            imgs, boxes = self.augment(imgs, boxes)
        if ret_orig:
            return np.array(imgs), boxes, np.array(imgs_orig), boxes_orig
        else:
            return np.array(imgs), boxes
    
    def resize_images(self, imgs, boxes):
        DEBUG = False
        w_n, h_n = self.out_size
        for i in range(len(imgs)):
            img_cur = imgs[i]
            if DEBUG: print(i)
            if DEBUG: print('Before ', img_cur.shape)
            s_f_y = img_cur.shape[1]/h_n
            s_f_x = img_cur.shape[0]/w_n
            if DEBUG: print(s_f_x, s_f_y)
            if s_f_x > 1 or s_f_y > 1:
                if s_f_y > s_f_x:
                    scale = img_cur.shape[0]/img_cur.shape[1]
                    img_cur = cv2.resize(img_cur, (int(h_n), int(scale*h_n)))
                    boxes[i] = np.array([[int(val/s_f_y) for val in z] for z in boxes[i]]).copy()
                else:
                    scale = img_cur.shape[1]/img_cur.shape[0]
                    img_cur = cv2.resize(img_cur, (int(scale*w_n), int(w_n)))
                    boxes[i] = np.array([[int(val/s_f_x) for val in z] for z in boxes[i]]).copy()                 
            if DEBUG: print('After ', img_cur.shape)
            y_pad = w_n - img_cur.shape[0]
            y_r = randint(0, y_pad)
            x_pad = h_n - img_cur.shape[1]
            x_r = randint(0, x_pad)
            if DEBUG: print('Padding', y_r, y_pad, x_r, x_pad)
            img_cur = cv2.copyMakeBorder(img_cur, y_r, y_pad - y_r, x_r, x_pad - x_r, cv2.BORDER_CONSTANT, value=(0,0,0))
            if DEBUG: print('Post border ', img_cur.shape)
            imgs[i] = img_cur.copy()
            boxes[i] = np.array([[z[0]+x_r, z[1]+y_r, z[2]+x_r, z[3]+y_r] for z in boxes[i]]).copy()
            # if boxes[i][2] > boxes[i][0]:
            #     raise ValueError('Something has gone wrong: ', boxes[i], x_r, y_r)
        return imgs, boxes
    
    def bbox_r(self, bbox):
        return [bbox.x1, bbox.y1, bbox.x2, bbox.y2]

    def augment(self, imgs, boxes):
        ia_bb = [ia.BoundingBoxesOnImage([ia.BoundingBox(x1 = i[0], y1 = i[1], x2 = i[2], y2 = i[3]) for i in boxes[n]], shape = imgs[n].shape) \
                for n in range(len(imgs))]
        seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.AddElementwise((-20, 20), per_channel=1)),
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.10*255))),
            iaa.Sometimes(0.5, iaa.Multiply((0.75, 1.25), per_channel=1)),
            iaa.Sometimes(0.5, iaa.MultiplyElementwise((0.75, 1.25))),
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 1.0))),
            iaa.Fliplr(0.5),
            iaa.Sometimes(0.95, iaa.SomeOf(1,
                [iaa.CoarseDropout(p=(0.10, 0.25), size_percent=(0.25, 0.5)),
                 iaa.CoarseDropout(p=(0.0, 0.15), size_percent=(0.1, 0.25)),
                 iaa.Dropout(p=(0, 0.25)),
                 iaa.CoarseSaltAndPepper(p=(0, 0.25), size_percent = (0.1, 0.2))])),
            iaa.Affine(scale = iap.Choice([iap.Uniform(0.4, 1), iap.Uniform(1, 3)]), rotate=(-180, 180))
        ])
        seq_det = seq.to_deterministic()
        image_b_aug = seq_det.augment_images(imgs)
        bbs_b_aug = seq_det.augment_bounding_boxes(ia_bb)
        bbs_b_aug = [b.remove_out_of_image().cut_out_of_image() for b in bbs_b_aug]
        return image_b_aug, [np.array([self.bbox_r(j) for j in i.bounding_boxes]) for i in bbs_b_aug]
    
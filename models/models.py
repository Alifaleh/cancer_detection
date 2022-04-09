# -*- coding: utf-8 -*-

from odoo import models, fields, api
from odoo.exceptions import UserError

import base64
import os
from glob import glob
import numpy as np
import numpy as np
import pydicom as dicom
import os
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import zipfile
import shutil
import math


workspace = f'{os.path.dirname(__file__)}/workspace'
lung_weights_file_path = f'{os.path.dirname(__file__)}/weight/lung.hdf5'
brain_weights_file_path = f'{os.path.dirname(__file__)}/weight/brain.h5'



class ResPartnerExt(models.Model):
    _inherit = 'res.partner'

    gender = fields.Selection([('male', 'Male'), ('female', 'Female')])
    age = fields.Integer()

class PartnerScan(models.Model):
    _name = 'partner.scan'
    _description = 'partner scan'
    _rec_name = 'partner_id'
    _inherit = ['mail.thread']

    # cancer type used in multi cancer
    cancer_type = fields.Selection([('lung', 'Lung'), ('brain', 'Brain')])

    partner_id = fields.Many2one('res.partner',string='Patient')
    file_type = fields.Selection([('npy', 'NPY'), ('dcm', 'DCM'), ('jpg', 'JPG')])
    scan_file = fields.Binary()
    classification = fields.Char(compute="_compute_classification",string = "Status", store=True, readonly=True)

    @api.onchange('scan_file')
    @api.constrains('scan_file')
    def _compute_classification(self):
        for rec in self:
            if rec.scan_file:

                scan_file = base64.b64decode(rec.scan_file)

                if rec.cancer_type == 'lung':
                    if rec.file_type == 'dcm':
                        open(f'{workspace}/input.zip', 'wb').write(scan_file)
                        with zipfile.ZipFile(f'{workspace}/input.zip', 'r') as zip_ref:
                            zip_ref.extractall(f'{workspace}/input')
                        os.remove(f'{workspace}/input.zip')
                        self.dcm2npy()
                        shutil.rmtree(f'{workspace}/input')
                    elif rec.file_type == 'npy':
                        open(f'{workspace}/input.npy', 'wb').write(scan_file)
                        
                    input_array = self.npy2nparray()
                    os.remove(f'{workspace}/input.npy')

                    model = self.build_lung_model()
                    prediction = self.get_lung_result(model, input_array)


                elif rec.cancer_type == 'brain':
                    if rec.file_type == 'dcm':
                        open(f'{workspace}/input.zip', 'wb').write(scan_file)
                        with zipfile.ZipFile(f'{workspace}/input.zip', 'r') as zip_ref:
                            zip_ref.extractall(f'{workspace}/input')
                        os.remove(f'{workspace}/input.zip')
                        self.dcm2jpg()
                        shutil.rmtree(f'{workspace}/input')

                    elif rec.file_type == 'jpg':
                        open(f'{workspace}/input.zip', 'wb').write(scan_file)
                        with zipfile.ZipFile(f'{workspace}/input.zip', 'r') as zip_ref:
                            zip_ref.extractall(f'{workspace}/input_jpg')
                        os.remove(f'{workspace}/input.zip')

                        
                    dg = self.jpg2dg()
                    model = self.build_brain_model()
                    prediction = self.get_brain_result(model, dg)
                    shutil.rmtree(f'{workspace}/input_jpg')

                
                rec.classification = prediction
            else:
                rec.classification = "No Npy File"


    def importing_data(self, path):
        path = '/usr/lib/python3/dist-packages/odoo/c-addons/cancer_detection/models/workspace/input_jpg/*.jpg'
        sample = []
        for filename in glob(path):
            #img = Image.open(filename,'r')
            #IMG = np.array(img)
            sample.append(filename)
        return sample


    def jpg2dg(self):
        target_size = [224, 224]
        datagen = ImageDataGenerator(rescale = 1./255)
        jpg_directory = f'{workspace}/input_jpg'

        test_path = f'{jpg_directory}/*.jpg'
        test_path_data = self.importing_data(test_path)
        df_test = pd.DataFrame({'image':test_path_data})

        dg = datagen.flow_from_dataframe(df_test,
                                            directory = f'{jpg_directory}/*.jpg',
                                            x_col = 'image',
                                            y_col = None,
                                            target_size = target_size,
                                            color_mode = 'grayscale',
                                            class_mode = None,
                                            batch_size = 10,
                                            shuffle = False,
                                            interpolation = 'bilinear')
        return dg


    def dcm2jpg(self):
        PNG = False
        folder_path = f'{workspace}/input'
        jpg_folder_path = f'{workspace}/input_jpg'
        images_path = os.listdir(folder_path)
        for n, image in enumerate(images_path):
            ds = dicom.dcmread(os.path.join(folder_path, image))
            pixel_array_numpy = ds.pixel_array
            if PNG == False:
                image = image.replace('.dcm', '.jpg')
            else:
                image = image.replace('.dcm', '.png')
            cv2.imwrite(os.path.join(jpg_folder_path, image), pixel_array_numpy)
            if n % 50 == 0:
                print('{} image converted'.format(n))

    def _compute_doctor(self):
        for rec in self:
            rec.doctor_id = rec.create_uid

    def npy2nparray(self):
        xtest=[]
        lll=glob(f'{workspace}/input.npy')
        for x in lll :
            xx=np.load(x)
            if xx.shape == (64,64,64):
                xtest.append(xx)
        xtest=np.array(xtest)
        return xtest


    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]


    def mean(self, a):
        return sum(a) / len(a)


    def dcm2npy(self):
        IMG_SIZE_PX=64
        SLICE_COUNT=64
        hm_slices=64
        data_dir = f'{workspace}/'
        patients = [filename for filename in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,filename))]
        for num,patient in enumerate(patients):
            path = data_dir + patient
            slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

            new_slices = []
            slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_SIZE_PX,IMG_SIZE_PX)) for each_slice in slices]
            chunk_sizes = math.ceil(len(slices) / hm_slices)
            for slice_chunk in self.chunks(slices, chunk_sizes):
                    slice_chunk = list(map(self.mean, zip(*slice_chunk)))
                    new_slices.append(slice_chunk)
                    xxxx=np.array(new_slices)
 
        yyyy=cv2.resize(xxxx,(64,64))    
        np.save(f'{workspace}/input.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), yyyy)    



#########################################################################
############################# Models ####################################
#########################################################################

########################### Lung cancer model ###########################

    def build_lung_model(self):
        img_rows = 64
        img_cols=64
        channels=64
        num_classes = 2
        middle_layers_activation = "relu"
        last_layer_activation = "softmax"
        INIT_LR = 0.01
        batch_size = 16
        epochs = 10
        input_shape = (img_rows, img_cols, channels)

        model = Sequential()
        model.add(Conv2D(256, (5, 5), padding="same", input_shape=input_shape))
        model.add(Activation(middle_layers_activation))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(512, (5, 5), padding="same"))
        model.add(Activation(middle_layers_activation))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   
        model.add(Flatten())
        model.add(Dense(1000))
        model.add(Activation(middle_layers_activation)) 
        model.add(Dropout(0.3))
        model.add(Dense(num_classes))
        model.add(Activation(last_layer_activation))
        opt = SGD(lr=INIT_LR, decay=INIT_LR / epochs)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
        model.load_weights(lung_weights_file_path)
        return model

    def get_lung_result(self, model, input_array):
        result = model.predict(input_array)
        if result [0][0] < result[0][1]:
            prediction = 'Cancer Detected'
        else: 
            prediction = 'Normal'
        return prediction


########################### Brain cancer model ###########################

    def build_brain_model(self):
        '''Sequential Model creation'''
        Cnn = Sequential()
        Cnn.add(Conv2D(64,(5,5), activation = 'relu', padding = 'same',strides=(2,2), input_shape = [224,224,1]))
        Cnn.add(MaxPooling2D(2))
        Cnn.add(Conv2D(128,(5,5), activation = 'relu', padding = 'same', strides=(2,2)))
        Cnn.add(Conv2D(128,(5,5), activation = 'relu', padding = 'same', strides=(2,2)))
        Cnn.add(Conv2D(256,(5,5), activation = 'relu', padding = 'same', strides=(2,2)))
        Cnn.add(MaxPooling2D(2))
        #Cnn.add(GlobalAveragePooling2D())
        Cnn.add(Flatten())
        Cnn.add(Dense(64, activation = 'relu'))
        Cnn.add(Dropout(0.4))
        Cnn.add(Dense(32, activation = 'relu'))
        Cnn.add(Dropout(0.4))
        Cnn.add(Dense(2, activation = 'softmax'))
        Cnn.load_weights(brain_weights_file_path)
        return Cnn

    def get_brain_result(self, model, dg):
        result = model.predict(dg)
        for pred in result:
            if pred[1] > pred[0]:
                prediction = 'Normal'
                return prediction
            else:
                continue

        prediction = 'Tumor Detected'
        return prediction

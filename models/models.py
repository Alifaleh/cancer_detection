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

workspace = f'{os.path.dirname(__file__)}/workspace'
weights_file_path = f'{os.path.dirname(__file__)}/weight/weights3d_lnet.hdf5'


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
    cancer_type = fields.Selection([('lung', 'Lung'), ('liver', 'Liver')])

    partner_id = fields.Many2one('res.partner',string='Patient')
    npy_file = fields.Binary()
    classification = fields.Char(compute="_compute_classification",string = "Status", store=True, readonly=True)

    @api.onchange('npy_file')
    @api.constrains('npy_file')
    def _compute_classification(self):
        for rec in self:
            if rec.npy_file:
                open(f'{workspace}/input.npy', 'wb').write(base64.b64decode(rec.npy_file))
                input_array = self.npy2nparray()
                os.remove(f'{workspace}/input.npy')

                # the following if statements used for multi cancer detection
                if rec.cancer_type == 'lung':
                    model = self.build_lung_model()
                    prediction = self.get_lung_result(model, input_array)
                elif rec.cancer_type == 'liver':
                    model = self.build_liver_model()
                    prediction = self.get_liver_result(model, input_array)

                rec.classification = prediction
            else:
                rec.classification = "No Npy File"

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
        model.load_weights(weights_file_path)
        return model

    def get_lung_result(self, model, input_array):
        result = model.predict(input_array)
        if result [0][0] < result[0][1]:
            prediction = 'Cancer Detected'
        else: 
            prediction = 'Normal'
        return prediction


########################### Liver cancer model ###########################

    def build_liver_model(self):
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
        model.load_weights(weights_file_path)
        return model

    def get_liver_result(self, model, input_array):
        result = model.predict(input_array)
        if result [0][0] < result[0][1]:
            prediction = 'Cancer Detected'
        else: 
            prediction = 'Normal'
        return prediction

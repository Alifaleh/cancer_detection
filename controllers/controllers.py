# -*- coding: utf-8 -*-
# from odoo import http


# class CancerDetection(http.Controller):
#     @http.route('/cancer_detection/cancer_detection/', auth='public')
#     def index(self, **kw):
#         return "Hello, world"

#     @http.route('/cancer_detection/cancer_detection/objects/', auth='public')
#     def list(self, **kw):
#         return http.request.render('cancer_detection.listing', {
#             'root': '/cancer_detection/cancer_detection',
#             'objects': http.request.env['cancer_detection.cancer_detection'].search([]),
#         })

#     @http.route('/cancer_detection/cancer_detection/objects/<model("cancer_detection.cancer_detection"):obj>/', auth='public')
#     def object(self, obj, **kw):
#         return http.request.render('cancer_detection.object', {
#             'object': obj
#         })

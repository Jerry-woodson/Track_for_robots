# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.utile_tctrackplus.loss import select_cross_entropy_loss, weight_l1_loss,l1loss,IOULoss,DISCLE
from pysot.models.backbone.temporalbackbonev2 import TemporalAlexNet

from pysot.model.backbone.fpn import Feature
from pysot.models.utile_tctrackplus.utile import APN
from pysot.models.utile_tctrackplus.utiletest import APNtest
import matplotlib.pyplot as plt

import numpy as np



class ModelBuilder_tctrackplus(nn.Module):
    def __init__(self,label):
        super(ModelBuilder_tctrackplus, self).__init__()

        input = t.rand(1, 3, 480, 640)
        #self.backbone = TemporalAlexNet().cuda()
        self.backbone = Feature(input).cuda()
        
        if label=='test':
            self.grader=APNtest(cfg).cuda()
        else:
            self.grader=APN(cfg).cuda()
        self.cls3loss=nn.BCEWithLogitsLoss()
        self.IOULOSS=IOULoss()

    def template(self, z,x):
        with t.no_grad():
            zf,_,_ = self.backbone.init(z)
            self.zf=zf

            xf,xfeat1,xfeat2 = self.backbone.init(x)   
            
            ppres=self.grader.conv1(self.xcorr_depthwise(xf,zf))

            self.memory=ppres
            self.featset1=xfeat1
            self.featset2=xfeat2
            

    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
    
    def track(self, x):
        with t.no_grad():
            
            xf,xfeat1,xfeat2 = self.backbone.eachtest(x,self.featset1,self.featset2)  
                        
            loc,cls2,cls3,memory=self.grader(xf,self.zf,self.memory)
                        
            self.memory=memory
            self.featset1=xfeat1
            self.featset2=xfeat2
            
        return {
                'cls2': cls2,
                'cls3': cls3,
                'loc': loc
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)

        return cls


    def getcentercuda(self,mapp):

        def dcon(x):
           x[t.where(x<=-1)]=-0.99
           x[t.where(x>=1)]=0.99
           return (t.log(1+x)-t.log(1-x))/2 
        
        size=mapp.size()[3]
        #location 
        x=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63)-287//2,size).reshape(-1)).cuda()
        y=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63).reshape(-1,1)-287//2,size).reshape(-1)).cuda()
        
        shap=dcon(mapp)*143
        
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))


        w=shap[:,0,yy,xx]+shap[:,1,yy,xx]
        h=shap[:,2,yy,xx]+shap[:,3,yy,xx]
        x=x-shap[:,0,yy,xx]+w/2+287//2
        y=y-shap[:,2,yy,xx]+h/2+287//2

        anchor=t.zeros((cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,size**2,4)).cuda()

        anchor[:,:,0]=x-w/2
        anchor[:,:,1]=y-h/2
        anchor[:,:,2]=x+w/2
        anchor[:,:,3]=y+h/2
        return anchor
    

    
    def forward(self,data,videorange):
        """ only used in training
        """

        presearch=data['pre_search'].cuda()
        template = data['template'].cuda()
        search =data['search'].cuda()
        bbox=data['bbox'].cuda()
        labelcls2=data['label_cls2'].cuda()
        labelxff=data['labelxff'].cuda()
        labelcls3=data['labelcls3'].cuda()
        weightxff=data['weightxff'].cuda()
        


        
        presearch=t.cat((presearch[:,cfg.TRAIN.videorangemax-videorange:,:,:,:],search.unsqueeze(1)),1)    
            
        zf = self.backbone(template.unsqueeze(1))
        
        xf = self.backbone(presearch) ###b l c w h
        xf=xf.view(cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,videorange+1,xf.size(-3),xf.size(-2),xf.size(-1))
        
        loc,cls2,cls3=self.grader(xf[:,-1,:,:,:],zf,xf[:,:-1,:,:,:].permute(1,0,2,3,4))

       
        cls2 = self.log_softmax(cls2) 

        
 
        cls_loss1 = select_cross_entropy_loss(cls2, labelcls2)
        cls_loss2 = self.cls3loss(cls3, labelcls3)  
        
        pre_bbox=self.getcentercuda(loc) 
        bbo=self.getcentercuda(labelxff) 
        
        loc_loss1=self.IOULOSS(pre_bbox,bbo,weightxff)
        loc_loss2=weight_l1_loss(loc,labelxff,weightxff)
        loc_loss3=DISCLE(pre_bbox,bbo,weightxff)
        loc_loss=cfg.TRAIN.w1*loc_loss1+cfg.TRAIN.w2*loc_loss2+cfg.TRAIN.w3*loc_loss3

        
        cls_loss=cfg.TRAIN.w4*cls_loss1+cfg.TRAIN.w5*cls_loss2

        

        outputs = {}
        outputs['total_loss'] =\
            cfg.TRAIN.LOC_WEIGHT*loc_loss\
                +cfg.TRAIN.CLS_WEIGHT*cls_loss
                
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss1'] = loc_loss1
        outputs['loc_loss2'] = loc_loss2
        outputs['loc_loss3'] = loc_loss3
                                                    #2 4 1  都用loss2

        return outputs

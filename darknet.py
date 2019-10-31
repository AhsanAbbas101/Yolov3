# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:05:42 2019

@author: Ahsan
"""

# Detector



import torch
import torch.nn as nn
from util import bbox_iou

#from torchsummary import summary

def conv_block(in_f, out_f, kernel_size, stride , padding):
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=kernel_size, stride=stride , padding= (kernel_size-1)//2 , bias=False),
            nn.BatchNorm2d(out_f),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Sequential(
                conv_block(in_f=self.out_channels, out_f=self.in_channels , kernel_size=1, stride=1, padding=1),
                conv_block(in_f=self.in_channels, out_f=self.out_channels, kernel_size=3, stride=1, padding=1)
                )
        
        self.activate = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
    
    def forward(self, x):
               
        residual = x
        #if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class YOLOLayer(nn.Module):
    
    def __init__(self,inp_dim, anchors, num_classes, device = 'cpu'):
        super().__init__()
        
        
        self.anchors = anchors
        self.num_classes = num_classes
        self.inp_dim = inp_dim
        
        self.ignore_threshold = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100
        self.grid_size = 0
        self.device = device
        
        
    def forward(self, x , target=None, img_dim = None):
        
        prediction = x
        #self.inp_dim = img_dim
        #[1, 255 , 52 ,52 ]
        ByteTensor = torch.cuda.ByteTensor if prediction.is_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor
        
        batch_size = prediction.size(0)
        stride =  self.inp_dim // prediction.size(2)
        grid_size = self.inp_dim // stride
        bbox_attrs = 5 + self.num_classes
        num_anchors = len(self.anchors)
        
        # Reshape Output
        prediction = (
                prediction.view(batch_size, num_anchors, bbox_attrs, grid_size, grid_size) #  [ 1, 3, 52, 52 , 85]
                .permute(0,1,3,4,2)
                .contiguous()
                )
        
        #Sigmoid the  centre_X, centre_Y, object confidencce and class pred.
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        
        
        #prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
        #prediction = prediction.transpose(1,2).contiguous()
        #prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
        
        
        # 0 - CenterX , 1 - CenterY , 2 - Height, 3 - Width , 4 - confidence , 5... class pred
        #prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
        #prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
        #prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
        #prediction[:,:,5: 5 + self.num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + self.num_classes]))
        
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size,CUDA=prediction.is_cuda)
        
        # Add Offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.x_offset
        pred_boxes[..., 1] = y.data + self.y_offset
        pred_boxes[..., 2] = torch.exp(w.data)* self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data)* self.anchor_h
        
        #prediction[:,:,:2] += self.x_y_offset
        #prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*self.scaled_anchors
        #prediction[:,:,:4] *= self.stride
        
        # TODO see device
        output = torch.cat(
                (
                    pred_boxes.view(batch_size, -1, 4) * stride,
                    pred_conf.view(batch_size, -1, 1),
                    pred_cls.view(batch_size, -1 , self.num_classes),
                ),
                -1,
                
        )
        
        if target is None:
            return output, 0
        else:
            # Function Call
          
            #Convert to position relative to box
            #target_boxes = target[:,2:6] * grid_size
            
            # [1,6]
            target_boxes = target[:,2:6]
            gxy = target_boxes[:, :2]
            gwh = target_boxes[:, 2:]
            
            
            
            # Get anchors with best IOU
            #print(grid_size)
            #print(self.anchors)
            #print(target_boxes.tolist())
            #print(torch.FloatTensor(self.anchors).size())
            #print(torch.FloatTensor(self.anchors).unsqueeze(0).size())
            #print(target_boxes.size())
            
            #x = [ print(torch.FloatTensor(anchor), target_boxes) for anchor in self.anchors ]
            #print(x)
            #for anchor in self.anchors:
            #    print (torch.FloatTensor(anchor))
            # Get anchors with best IOU
            
            ious = torch.stack([ bbox_iou(FloatTensor(anchor).unsqueeze(0), gwh , False) for anchor in self.anchors ])
            best_ious , best_n = ious.max(0)
            
            # Separate target_labels
            b, target_labels = target[:, :2].long().t()
            gx , gy = gxy.t() # not scaled
            gw , gh = gwh.t() # not scaled
            gi , gj = gxy.long().t()% grid_size
            
            
            
            # Set masks 
            obj_mask = ByteTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)
            noobj_mask = ByteTensor(batch_size, num_anchors, grid_size, grid_size).fill_(1)


            #print (b, best_n, gj.size(), gi.size())

            obj_mask[b, best_n, gj , gi] = 1
            noobj_mask[b, best_n, gj , gi] = 0
            
            
            # Set noobj mask to zero where iou exceeds ignore threshold
            for i, anchor in enumerate(ious.t()):
                noobj_mask[b[i], anchor> self.ignore_threshold , gj[i], gi[i]] = 0
            
            tx =  FloatTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)
            ty =  FloatTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)
            tw =  FloatTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)
            th =  FloatTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)
            class_mask =  FloatTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)
            #iou_scores =  FloatTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)
            tcls =  FloatTensor(batch_size, num_anchors, grid_size, grid_size, self.num_classes).fill_(0)
            
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gx.floor()
            ty[b, best_n, gj, gi] = gy - gy.floor()
            # Width and Height
            
            #print (self.scaled_anchors.size())
            #self.scaled_anchors = [(a[0]/self.stride, a[1]/self.stride) for a in self.anchors]
            #self.scaled_anchors = torch.FloatTensor(self.scaled_anchors)
            
            tw[b, best_n, gj, gi] = torch.log(gw/ self.scaled_anchors[best_n][:, 0] + 1e-16)
            th[b, best_n, gj, gi] = torch.log(gh/ self.scaled_anchors[best_n][:, 1] + 1e-16)
            # One Hot encoding of label
            tcls[b, best_n, gj, gi, target_labels] = 1
            
            #print(target_labels.size())
            #print(prediction[:,:,5:5+self.num_classes].argmax(-1))
            
            # Compute labels correctness and iou at best anchor
            class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
            #iou_scores[b, best_n, gj, gi] = bbox_iou(prediction[:,:,:4, ], target_boxes) # --- Masla
            
            tconf = obj_mask.float()
            
            
            # Compute Loss
            MSEloss = nn.MSELoss()
            BCEloss = nn.BCELoss()
            
            #print("OBject Mask")
            #print(obj_mask.size())
            #prediction.view(batch_size)
            #print("Prediction: ",prediction[:,:,0][obj_mask].size())
            #print("tx ", tx[obj_mask].size())
            
            loss_x = MSEloss( x[obj_mask] , tx[obj_mask])
            loss_y = MSEloss( y[obj_mask] , ty[obj_mask])
            loss_w = MSEloss( w[obj_mask] , tw[obj_mask])
            loss_h = MSEloss( h[obj_mask] , th[obj_mask])
            loss_conf_obj = BCEloss( pred_conf[obj_mask] , tconf[obj_mask])
            loss_conf_noobj = BCEloss( pred_conf[noobj_mask] , tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = BCEloss( pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        
            return output, total_loss

    def compute_grid_offsets(self, grid_size, CUDA=True):
        """
            Generates Offsets in self.x_y_offset
            Scales Anchors with stride in self.scaled_anchors
        """
        self.grid_size = grid_size
        self.stride = self.inp_dim // self.grid_size

        FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


        self.x_offset = torch.arange(grid_size).repeat(grid_size, 1).view([1,1,grid_size,grid_size]).type(FloatTensor)
        self.y_offset = torch.arange(grid_size).repeat(grid_size, 1).t().view([1,1,grid_size,grid_size]).type(FloatTensor)
        
        
        self.scaled_anchors = FloatTensor([(a[0]/self.stride, a[1]/self.stride) for a in self.anchors])

        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, len(self.anchors), 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, len(self.anchors), 1, 1))
        


class DarkNet(nn.Module):
    
    
    
    def __init__(self, target=None):
        super().__init__()
        
        self.inp_dim = 416
        self.num_classes = 80
        self.nBoxes = 3
        
        
        self.anchors_big = [(116,90),(156,198),(373,326)]
        self.anchors_medium = [(30,61),(62,45),(59,119)]
        self.anchors_small = [(10,13),(16,30),(33,23)]
        
        self.target = target
        
        # Create DarkNet Modules 
        
        # Residual Block
        # Convo Layers
        
        # Call 
        
        self.block_A = nn.Sequential(
                conv_block(in_f=3, out_f=32, kernel_size=3, stride=1, padding=1 ),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                conv_block(in_f=32, out_f=64, kernel_size=3, stride=2, padding=1)
                )

        self.block_B = nn.Sequential(
                ResidualBlock(32,64),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                conv_block(in_f=64, out_f=128, kernel_size=3, stride=2, padding=1 )
                )

        self.block_C = nn.Sequential(
                ResidualBlock(64,128),
                ResidualBlock(64,128),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                conv_block(in_f=128, out_f=256, kernel_size=3, stride=2, padding=1 )
                )

        self.block_D_1 = nn.Sequential(
                ResidualBlock(128,256),ResidualBlock(128,256),ResidualBlock(128,256),ResidualBlock(128,256),
                ResidualBlock(128,256),ResidualBlock(128,256),ResidualBlock(128,256),ResidualBlock(128,256)
                )
        self.block_D_2 = nn.Sequential(
                #nn.MaxPool2d(kernel_size=2, stride=2),
                conv_block(in_f=256, out_f=512, kernel_size=3, stride=2, padding=1 )
                )
        
        self.block_E_1 = nn.Sequential(
                ResidualBlock(256,512),ResidualBlock(256,512),ResidualBlock(256,512),ResidualBlock(256,512),
                ResidualBlock(256,512),ResidualBlock(256,512),ResidualBlock(256,512),ResidualBlock(256,512)
                )
        self.block_E_2 = nn.Sequential(
                #nn.MaxPool2d(kernel_size=2, stride=2),
                conv_block(in_f=512, out_f=1024, kernel_size=3, stride=2, padding=1 )
                )
        
        
        self.block_F = nn.Sequential(
                ResidualBlock(512,1024),ResidualBlock(512,1024),ResidualBlock(512,1024),ResidualBlock(512,1024),
                conv_block(in_f=1024, out_f=512, kernel_size=1, stride=1, padding=1 ),
                conv_block(in_f=512, out_f=1024, kernel_size=3, stride=1, padding=1 ),
                
                conv_block(in_f=1024, out_f=512, kernel_size=1, stride=1, padding=1 ),
                conv_block(in_f=512, out_f=1024, kernel_size=3, stride=1, padding=1 ),
                
                conv_block(in_f=1024, out_f=512, kernel_size=1, stride=1, padding=1 ),
                conv_block(in_f=512, out_f=1024, kernel_size=3, stride=1, padding=1 )
            
                )



        self.yolo_big = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=(self.nBoxes*(5+self.num_classes)), kernel_size=1, stride=1 , padding=(1-1)//2),
                # Activation - Linear
                nn.ReLU(inplace=True)
                
                
                )
        # Detector
        self.yolo_big_detector = YOLOLayer(self.inp_dim, self.anchors_big , self.num_classes )
        
        # yolo medium scale
        self.block_G = nn.Sequential(
                conv_block(in_f=1024, out_f=256, kernel_size=1, stride=1, padding=1 ),
                nn.Upsample(scale_factor = 2, mode = "bilinear")
                
                )
        self.block_H = nn.Sequential(
                conv_block(in_f=768, out_f=256, kernel_size=1, stride=1, padding=1 ),
                conv_block(in_f=256, out_f=512, kernel_size=3, stride=1, padding=1 ),
                
                conv_block(in_f=512, out_f=256, kernel_size=1, stride=1, padding=1 ),
                conv_block(in_f=256, out_f=512, kernel_size=3, stride=1, padding=1 ),
                
                conv_block(in_f=512, out_f=256, kernel_size=1, stride=1, padding=1 ),
                conv_block(in_f=256, out_f=512, kernel_size=3, stride=1, padding=1 )
             
                )
        self.yolo_medium = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=(self.nBoxes*(5+self.num_classes)), kernel_size=1, stride=1 , padding=(1-1)//2),
                # Activation - Linear
                nn.ReLU(inplace=True)
                # Detector
                
                )
        self.yolo_medium_detector = YOLOLayer(self.inp_dim, self.anchors_medium , self.num_classes )
        
        # yolo small scale
        self.block_I = nn.Sequential(
                conv_block(in_f=512, out_f=128, kernel_size=1, stride=1, padding=1),
                nn.Upsample(scale_factor = 2, mode = "bilinear")
                )
        
        self.yolo_small = nn.Sequential(
                conv_block(in_f=384, out_f=128, kernel_size=1, stride=1, padding=1),
                conv_block(in_f=128, out_f=256, kernel_size=3, stride=1, padding=1),
                
                conv_block(in_f=256, out_f=128, kernel_size=1, stride=1, padding=1),
                conv_block(in_f=128, out_f=256, kernel_size=3, stride=1, padding=1),
                
                conv_block(in_f=256, out_f=128, kernel_size=1, stride=1, padding=1),
                conv_block(in_f=128, out_f=256, kernel_size=3, stride=1, padding=1),
                
                nn.Conv2d(in_channels=256, out_channels=(self.nBoxes*(5+self.num_classes)), kernel_size=1, stride=1 , padding=(1-1)//2),
                # Activation - Linear
                nn.ReLU(inplace=True)
                
                )
        #Detector
        self.yolo_small_detector = YOLOLayer(self.inp_dim, self.anchors_small , self.num_classes)
        
        
    def forward(self,x):
        
        x = self.block_A(x)
        x = self.block_B(x)
        x = self.block_C(x)
        
        x_d_1 = self.block_D_1(x)
        x = self.block_D_2(x_d_1)
        
        x_e_1 = self.block_E_1(x)
        x = self.block_E_2(x_e_1)
        
        x_f = self.block_F(x)
        
        # yolo_big
        x_big = self.yolo_big(x_f)
        x_big, loss_1 = self.yolo_big_detector(x_big, self.target)
        
        #x_big = x_big.data
        #x_big = predict_transform(x_big, self.inp_dim, self.anchors_big , self.num_classes, self.CUDA)
        
        x = self.block_G(x_f)
        

        
        # Concat E1 - G
        x = torch.cat((x_e_1,x),1)
        
        #print(x.size())
        
        x_h = self.block_H(x)
        
        # yolo_med
        x_med = self.yolo_medium(x_h)
        x_med, loss_2 = self.yolo_medium_detector(x_med, self.target)
        
        #x_med = x_med.data
        #x_med = predict_transform(x_med, self.inp_dim, self.anchors_medium, self.num_classes, self.CUDA)
        
        
        x = self.block_I(x_h)
        
        # Concat D1- I
        x = torch.cat((x_d_1,x),1)

        # yolo_small
        x_small = self.yolo_small(x)
        x_small, loss_3 = self.yolo_small_detector(x_small, self.target)
        #x_small = x_small.data
        #x_small = predict_transform(x_small, self.inp_dim, self.anchors_small, self.num_classes, self.CUDA)
        
        
        #return torch.cat((x_big.data,x_med.data,x_small.data),1)
        return torch.cat((x_big.data,x_med.data,x_small.data),1) , (loss_1+loss_2+loss_3)

# Yolov3 - Darknet-53 - Architecture
"""
x = torch.rand(1,3,416,416)
x *= 255
target = torch.zeros(1,6)
print(target.size())
target[0,0] = 0
target[0,2] = 333
target[0,3] = 72
target[0,4] = 425
target[0,5] = 158
target[0,1] = 0
print(target)
model = DarkNet(target)
model.forward(x)
#summary(model, (3,416,416))
"""





















# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:05:42 2019

@author: Ahsan
"""

# Detector



import torch
import torch.nn as nn

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
    
    def __init__(self,inp_dim, anchors, num_classes, CUDA = True):
        super().__init__()
        
        
        self.anchors = anchors
        self.num_classes = num_classes
        self.inp_dim = inp_dim
        
        self.ignore_threshold = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100
        self.grid_size = 0
        
        
    def forward(self, x , target=None, img_dim = None):
        
        prediction = x
        #self.inp_dim = img_dim
        
        batch_size = prediction.size(0)
        stride =  self.inp_dim // prediction.size(2)
        grid_size = self.inp_dim // stride
        bbox_attrs = 5 + self.num_classes
        num_anchors = len(self.anchors)
        
        # Reshape Output
        prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
        prediction = prediction.transpose(1,2).contiguous()
        prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
        
        #Sigmoid the  centre_X, centre_Y, object confidencce and class pred.
        # 0 - CenterX , 1 - CenterY , 2 - Height, 3 - Width , 4 - confidence , 5... class pred
        prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
        prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
        prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
        prediction[:,:,5: 5 + self.num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + self.num_classes]))
        
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size,CUDA=x.is_cuda)
        
        # Add Offset and scale with anchors
        prediction[:,:,:2] += self.x_y_offset
        prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*self.scaled_anchors
        prediction[:,:,:4] *= self.stride
        
        
        if target is None:
            return prediction
        else:
            # Function Call
            
            #Convert to position relative to box
            target_boxes = target[:,2:6] * grid_size
            gxy = target_boxes[:, :2]
            gwh = target_boxes[:, 2:]
            
            # Get anchors with best IOU
            ious = torch.stack([ fuction_iou(anchor, gwh) for anchor in anchors ])
            best_ious , best_n = ious.max(0)
            
            # Separate target_labels
            b, target_labels = target[:, :2].long().t()
            gx , gy = gxy.t()
            gw , gh = gwh.t() 
            gi , gj = gxy.long().t()
            
            # Set masks 
            obj_mask[b, best_n, gj , gi] = 1
            noobj_mask[b, best_n, gj , gi] = 0
            
            # Set noobj mask to zero where iou exceeds ignore threshold
            for i, anchor in enumerate(ious.t()):
                noobj_mask[b[i], anchor> ignore_thres , gj[i], gi[i]] = 0
                
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gx.floor()
            ty[b, best_n, gj, gi] = gy - gy.floor()
            # Width and Height
            tw[b, best_n, gj, gi] = torch.log(gw/ anchors[best_n][:, 0] + 1e-16)
            th[b, best_n, gj, gi] = torch.log(gh/ anchors[best_n][:, 1] + 1e-16)
            # One Hot encoding of label
            tcls[b. best_n. gj, gi, target_labels] = 1
            
            
            # Compute labels correctness and iou at best anchor
            class_mask[b, best_n, gj, gi] = (prediction[:,:,5:5+self.num_classes].argmax(-1) == target_labels).float()
            iou_scores[b, best_n, gj, gi] = fuction_iou(prediction[:,:,:4, ], target_boxes) # --- Masla
            
            tconf = obj_mask.float()
            
            
            # Compute Loss
            MSEloss = nn.MSELoss()
            BCEloss = nn.BCELoss()
            loss_x = MSEloss( prediction[:,:,0][obj_mask] , tx[obj_mask])
            loss_y = MSEloss( prediction[:,:,1][obj_mask] , ty[obj_mask])
            loss_w = MSEloss( prediction[:,:,2][obj_mask] , tw[obj_mask])
            loss_h = MSEloss( prediction[:,:,3][obj_mask] , th[obj_mask])
            loss_conf_obj = BCEloss( prediction[:,:,4][obj_mask] , tconf[obj_mask])
            loss_conf_noobj = BCEloss( prediction[:,:,4][noobj_mask] , tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = BCEloss( prediction[:,:,5:5+self.num_classes][obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        
            return prediction, total_loss

    def compute_grid_offsets(self, grid_size, CUDA=True):
        """
            Generates Offsets in self.x_y_offset
            Scales Anchors with stride in self.scaled_anchors
        """
        self.grid_size = grid_size
        self.stride = self.inp_dim // self.grid_size
        
        grid = np.arange(grid_size)
        a,b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1,1)
        y_offset = torch.FloatTensor(b).view(-1,1)
        
        if CUDA:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        self.x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,len(self.anchors)).view(-1,2).unsqueeze(0)
        
        self.scaled_anchors = [(a[0]/self.stride, a[1]/self.stride) for a in self.anchors]
        self.scaled_anchors = torch.FloatTensor(self.scaled_anchors)
        self.scaled_anchors = self.scaled_anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
        
        if CUDA:
            self.scaled_anchors = self.scaled_anchors.cuda()
            
            
        

class DarkNet(nn.Module):
    
    
    
    def __init__(self):
        super().__init__()
        
        self.inp_dim = 416
        self.num_classes = 80
        self.nBoxes = 3
        self.CUDA = False
        
        self.anchors_big = [(116,90),(156,198),(373,326)]
        self.anchors_medium = [(30,61),(62,45),(59,119)]
        self.anchors_small = [(10,13),(16,30),(33,23)]
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
                nn.ReLU(inplace=True),
                # Detector
                YOLOLayer(self.inp_dim, self.anchors_big , self.num_classes , self.CUDA)
                )
        
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
                nn.ReLU(inplace=True),
                # Detector
                YOLOLayer(self.inp_dim, self.anchors_medium , self.num_classes , self.CUDA)
                )
        
        
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
                nn.ReLU(inplace=True),
                #Detector
                YOLOLayer(self.inp_dim, self.anchors_small , self.num_classes , self.CUDA)
                )
        
        
        
        
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
        
        #x_big = x_big.data
        #x_big = predict_transform(x_big, self.inp_dim, self.anchors_big , self.num_classes, self.CUDA)
        
        x = self.block_G(x_f)
        

        
        # Concat E1 - G
        x = torch.cat((x_e_1,x),1)
        
        print(x.size())
        
        x_h = self.block_H(x)
        
        # yolo_med
        x_med = self.yolo_medium(x_h)
        
        #x_med = x_med.data
        #x_med = predict_transform(x_med, self.inp_dim, self.anchors_medium, self.num_classes, self.CUDA)
        
        
        x = self.block_I(x_h)
        
        # Concat D1- I
        x = torch.cat((x_d_1,x),1)

        # yolo_small
        x_small = self.yolo_small(x)
        
        #x_small = x_small.data
        #x_small = predict_transform(x_small, self.inp_dim, self.anchors_small, self.num_classes, self.CUDA)
        
        
        #return torch.cat((x_big.data,x_med.data,x_small.data),1)
        return x   

# Yolov3 - Darknet-53 - Architecture

#model = DarkNet()
#summary(model, (3,416,416))






















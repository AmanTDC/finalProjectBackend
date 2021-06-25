import torch #works on tensors (not on numpy arrays)
import numpy as np  
import cv2 #for image processing tasks like reading an image from directory etc.
import boto3
import torch.nn as nn #nn implies neural networks
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from io import BytesIO

def conv3x3x3(in_planes, out_planes, stride=1):  #convolutional layer 3d for our model
    # 3x3x3 convolution with padding ##3d coonv layer is used as we are working with videos 
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        #self.conv1 = nn.Conv3d(
        #    3,
        #    64,
        #    kernel_size=(3,7,7),
        #    stride=(1, 2, 2),
        #    padding=(1, 3, 3),
        #    bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        #last_duration = 1
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):  #make yout data travel from one layer to another (feedforward)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")



def resnext101(**kwargs):
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model
no_to_gest = ['Doing_other_things','Drumming_Fingers','No_gesture','Pulling_Hand_In','Pulling_Two_Fingers_In',
    'Pushing_Hand_Away','Pushing_Two_Fingers_Away','Rolling_Hand_Backward','Rolling_Hand_Forward','Shaking_Hand',
    'Sliding_Two_Fingers_Down','Sliding_Two_Fingers_Left','Sliding_Two_Fingers_Right','Sliding_Two_Fingers_Up',
    'Stop_Sign','Swiping_Down','Swiping_Left','Swiping_Right','Swiping_Up','Thumb_Down','Thumb_Up','Turning_Hand_Clockwise',
    'Turning_Hand_Counterclockwise','Zooming_In_With_Full_Hand','Zooming_In_With_Two_Fingers','Zooming_Out_With_Full_Hand',
    'Zooming_Out_With_Two_Fingers']


model = resnext101(sample_size=100,sample_duration=12,num_classes=27)#25 gestures and 2 non gestures
s3 = boto3.client('s3')
res = s3.get_object(Bucket = 'myfinalyearproject',Key=f'model.pth')
state = torch.load(BytesIO(res["Body"].read()),map_location=torch.device('cpu'))
#dicto = torch.load("model.pth")
model.load_state_dict(state)
imgs = np.array(np.transpose(cv2.imread("100027//00001.jpg"),(2,0,1))).reshape(3,1,100,176)
for i in range(2,38,3):
    if(i<=9):
        toappend = "0"+str(i)
    else:
        toappend = str(i)
    imgs = np.column_stack((imgs,(np.transpose(cv2.imread("100027"+"//000"+toappend+".jpg"),(2,0,1))).reshape(3,1,100,176)))
    imgs = np.array(imgs)
        #print(imgs.shape)
imgs1 = imgs.reshape(1,1,3,13,100,176)
def  predict(imgs2):
    #folder = "10021"

    

    # #need imgs2 which is of shape 1,1,3,13,100,176
    # imgs = np.array(np.transpose(cv2.imread(folder+"//00001.jpg"),(2,0,1))).reshape(3,1,100,176)
    
    # for i in range(2,38,3):
    #     if(i<=9):
    #         toappend = "0"+str(i)
    #     else:
    #         toappend = str(i)
        

    
    #     imgs = np.column_stack((imgs,(np.transpose(cv2.imread(folder+"//000"+toappend+".jpg"),(2,0,1))).reshape(3,1,100,176)))
    #     imgs = np.array(imgs)
    
    
    # imgs2 = imgs.reshape(1,1,3,13,100,176)
    # #need end
    
    input_ = np.column_stack((imgs1,imgs2)).reshape(2,3,13,100,176)


    result = model(torch.tensor(input_[:,:,:,:,:121]).float())

    max = 0
    for i in range(len(result[0])):
        if result[1][i] > result[1][max]:
            max = i
    return (no_to_gest[max])
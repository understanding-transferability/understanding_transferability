from utils import *
from data import *
from logger import *
from scheduler import *
from wheel import *
from visualization import *
from gpuutils import *

def skip(data, label, is_train):
    return None
def transform(data, label, is_train):
    label = one_hot(102, label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label
batch_size = 32
ds = FileListDataset('food_train.txt', '', transform=transform, skip_pred=skip, is_train=True, imsize=256, auto_weight = True)
source_train = CustomDataLoader(ds, batch_size=batch_size, num_threads=2)

selectGPUs(1)
log = Logger('log/loss_surfacetest', clear=True)

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

resnet_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, "resnet101":models.resnet101, "resnet152":models.resnet152}

class ResNetFc(BaseFeatureExtractor):
    def __init__(self, model_name='resnet50',model_path=None, normalize=True):
        super(ResNetFc, self).__init__()
        self.model_resnet = resnet_dict[model_name](pretrained=False)
        if not os.path.exists(model_path):
            model_path = None
            print('invalid model path!')
        if model_path:
            self.model_resnet.load_state_dict(torch.load(model_path))
        if model_path or normalize:
            self.normalize = True
            self.mean = False
            self.std = False
        else:
            self.normalize = True

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.__in_features = model_resnet.fc.in_features

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def output_num(self):
        return self.__in_features

class ResNetFc1(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self, model_name='resnet50',model_path=None, normalize=True):
        super(ResNetFc1, self).__init__()
        self.model_resnet = resnet_dict[model_name](pretrained=False)
        if not os.path.exists(model_path):
            model_path = None
            print('invalid model path!')
        if model_path:
            self.model_resnet.load_state_dict(torch.load(model_path))
        if model_path or normalize:
            self.normalize = True
            self.mean = False
            self.std = False
        else:
            self.normalize = True

        model_resnet = self.model_resnet
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features
    
    
class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(
                self.bottleneck,
                nn.Sequential(
                    nn.BatchNorm1d(bottle_neck_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    self.fc
                ),
                nn.Softmax(dim=-1)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

feature_extractor = ResNetFc(model_name='resnet50',model_path='yourresnetpath').cuda()
feature_extractor1 = ResNetFc1(model_name='resnet50',model_path='yourresnetpath').cuda()
feature_extractor2 = ResNetFc1(model_name='resnet50',model_path='yourresnetpath').cuda()
cls = CLS(feature_extractor1.output_num(), 102, bottle_neck_dim=256).cuda()
net = nn.Sequential(feature_extractor, feature_extractor1, cls)

scheduler = lambda step, initial_lr : twostage(step, initial_lr, gamma=0.1, max_iter=39540)
def twostage(step, initial_lr, gamma = 0.1, max_iter = 6000):
    if step < max_iter:
        return initial_lr
    else:
        return gamma * initial_lr
optimizer_feature_extractor1 = OptimWithSheduler(optim.SGD(feature_extractor1.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9),
                            scheduler)
optimizer_cls = OptimWithSheduler(optim.SGD(cls.parameters(), lr=0.01, weight_decay=0.0005, momentum = 0.9),
                            scheduler)

epoch=0
#define the matrix of value along the trajectories
lossval = np.zeros((12000, 100)).astype('float32')

while epoch <499:
    for (i, ((im_source, label_source))) in enumerate(
            source_train.generator()):
        im_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        
        fs, feature_source, __, predict_prob_source = net(im_source)
        ce = CrossEntropyLoss(label_source, predict_prob_source)
          
        with OptimizerManager([optimizer_cls, optimizer_feature_extractor1]):
            loss =  ce 
            loss.backward()
        mmp = list(feature_extractor1.named_parameters())[72][-1].grad
        mmp = mmp.data
        for item in feature_extractor1.state_dict():
            feature_extractor2.state_dict()[item].copy_(feature_extractor1.state_dict()[item].data) 
        feature_extractor1.zero_grad()
        cls.zero_grad()
        #compute the loss value
        for pppi in range(100):
            feature_extractor2.state_dict()['model_resnet.layer3.0.conv1.weight'].copy_(list(feature_extractor2.named_parameters())[72][-1].data - 0.001 * mmp) 
            lossval[log.step, pppi] = (CrossEntropyLoss(label_source, cls(feature_extractor2(feature_extractor(im_source)))[-1]).cpu().data.numpy())
        
        log.step += 1

        if log.step % 10 == 1:
            counter = AccuracyCounter()
            counter.addOntBatch(variable_to_numpy(predict_prob_source[0: 256]), variable_to_numpy(label_source))
            acc_train = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce', 'acc_train'], globals())

        if log.step % 100 == 0:
            clear_output()
    epoch += 1
            

#save the figure
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10,5))
x=np.linspace(4000,12000,8000)
y=np.array(lossval[4000:12000,0])
yy = np.array(np.max(lossval[4000:12000,1:], axis = -1))

yyy = np.abs(yy - y)
yyyy = yyy + y
plt.plot(x,yyy,c='g',linewidth = '0.25',label = 'ImageNet pretrained')
plt.xlabel("Step",fontsize=18,)
plt.ylabel("Variation of loss",fontsize=18,)
plt.ylim(0,0.0008)
plt.legend(loc=0, numpoints=1,fontsize = 16)
plt.savefig('variation_of_loss.svg',format='svg',bbox_inches='tight', transparent=True,)
plt.show()
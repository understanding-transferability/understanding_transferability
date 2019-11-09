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
    data = tl.prepro.crop(data, 224, 224)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label

eta = 0.01
batch_size = 256

ds = FileListDataset('/food_train.txt', '', transform=transform, skip_pred=skip, is_train=True, imsize=256)
source_train = CustomDataLoader(ds, batch_size=batch_size, num_threads=2)
setGPU('7')

log = Logger('log/loss_surface2', clear=True)

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
            self.mean = (
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = (
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
            self.mean = (
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = (
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

im_source,label_source = source_train.generator().__next__()
lossval = np.zeros((200, 200)).astype('float32')

im_source = torch.from_numpy(im_source).cuda()
label_source = torch.from_numpy(label_source).cuda()

#calculate the two directions using filter normalization
vec1 = torch.from_numpy(np.random.randn(1048576).reshape(list(feature_extractor1.named_parameters())[147][-1].shape).astype('float32')).cuda()
vec1v = torch.sum(vec1 * vec1)
ranvec = torch.from_numpy(np.random.randn(1048576).reshape(list(feature_extractor1.named_parameters())[147][-1].shape).astype('float32')).cuda()
vec2  = ranvec - torch.sum(ranvec * vec1) * vec1 / vec1v
vec1 = vec1 / torch.sqrt(vec1v)
vec1norm = torch.norm(vec1.reshape(2048,512),dim = -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
filternorm = torch.norm(list(feature_extractor1.named_parameters())[147][-1].reshape(2048,512),dim = -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
vec2norm = torch.norm(vec2.reshape(2048,512),dim = -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
vec1 = vec1 / vec1norm * filternorm * eta
vec2 = vec2 / vec2norm * filternorm * eta
vec2 = vec2 - torch.sum(vec2* vec1) * vec1 / torch.sum(vec2 * vec2)

#calculate the loss landscapes 
feature_extractor.zero_grad()
feature_extractor1.zero_grad()
cls.zero_grad()
for x_ind in range(-100,100):
    for y_ind in range(-100,100):
        feature_extractor.zero_grad()
        feature_extractor1.zero_grad()
        cls.zero_grad()
        feature_extractor1.state_dict()['model_resnet.layer4.1.conv3.weight'].copy_(list(feature_extractor2.named_parameters())[147][-1].data +x_ind *  0.1 * vec1 + y_ind * 0.1 * vec2) 
        lossval[x_ind + 100, y_ind + 100] = (CrossEntropyLoss(label_source, cls(feature_extractor1(feature_extractor(im_source)))[-1]).cpu().data.numpy())

#save the loss landscape plot
np.save('loss_val.npy',lossval)
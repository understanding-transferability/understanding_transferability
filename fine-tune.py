from utils import *
from data import *
from logger import *
from scheduler import *
from wheel import *
from visualization import *
from gpuutils import *

def skip(data, label, is_train):
    return False

def transform(data, label, is_train):
    label = one_hot(200, label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label

batch_size = 32
ds = FileListDataset('../Transferable/cub200.txt', '', transform=transform, skip_pred=skip, is_train=True, imsize=256, auto_weight = True)
source_train = CustomDataLoader(ds, batch_size=batch_size, num_threads=2)
ds2 = FileListDataset('../Transferable/cub200_test.txt', '',transform=transform, skip_pred=skip, is_train=False, imsize=256)
target_test = CustomDataLoader(ds2, batch_size=batch_size, num_threads=2)
setGPU('0')
log = Logger('log/cub_scratchp', clear=True)

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
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
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

feature_extractor = ResNetFc(model_name='resnet50',model_path='yourresnetpath')
cls = CLS(feature_extractor.output_num(), 200, bottle_neck_dim=256)
net = nn.Sequential(feature_extractor, cls).cuda()
net = nn.DataParallel(net)

scheduler = lambda step, initial_lr : twostage(step, initial_lr, gamma = 0.1, max_iter =  18730)
def twostage(step, initial_lr, gamma = 0.1, max_iter = 6000):
    if step < max_iter:
        return initial_lr
    else:
        return gamma * initial_lr
optimizer_feature_extractor = OptimWithSheduler(optim.SGD(feature_extractor.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9),
                            scheduler)
optimizer_cls = OptimWithSheduler(optim.SGD(cls.parameters(), lr=0.01, weight_decay=0.0005, momentum = 0.9),
                            scheduler)

#build the list of pretrained parameters
params = list(feature_extractor.named_parameters())
para = []
for item in params:
    para.append(item[-1].cpu().data)
lslst=[]

#train
k=0
while k <200:
    for (i, ((im_source, label_source))) in enumerate(
            source_train.generator()):
        # =========================forward pass
        im_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        _, feature_source, __, predict_prob_source = net(im_source)
        ce = CrossEntropyLoss(label_source, predict_prob_source)
        deviation = 0
        #compute the deviation
        for i in range(53): 
            deviation+=  torch.norm(list(feature_extractor.named_parameters())[3*i][-1] - para[i*3].cuda())
        lslst.append(deviation.data.cpu().numpy())
        with OptimizerManager([optimizer_cls, optimizer_feature_extractor]):
            loss =  ce 
            loss.backward()
        log.step += 1

        if log.step % 10 == 1:
            counter = AccuracyCounter()
            counter.addOntBatch(variable_to_numpy(predict_prob_source[0: 256]), variable_to_numpy(label_source))
            acc_train = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce', 'acc_train','deviation'], globals())

        if log.step % 100 == 0:
            clear_output()
    k += 1
#test
with TrainingModeManager([feature_extractor,cls], train=False) as mgr, Accumulator(['predict_prob','predict_index','label']) as accumulator:
    for (i, (im, label)) in enumerate(target_test.generator()):
        im = Variable(torch.from_numpy(im), volatile=True).cuda()
        label = Variable(torch.from_numpy(label), volatile=True).cuda()
        ss, fs,_,  predict_prob = net.forward(im)
        predict_prob,label = [variable_to_numpy(x) for x in (predict_prob,label)]
        label = np.argmax(label, axis=-1).reshape(-1, 1)
        predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
        accumulator.updateData(globals())
        if i % 10 == 0:
        print(i)

for x in accumulator.keys():
    globals()[x] = accumulator[x]
    
#calculate accuracy
acc = float(np.sum(label.flatten() == predict_index.flatten()) )/ label.flatten().shape[0]
print(acc)

#plot the change of deviation
from matplotlib import pyplot
pyplot.plot(lslst)
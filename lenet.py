from utils import *
from data import *
from logger import *
from scheduler import *
from wheel import *
from visualization import *
from gpuutils import *
from collections import Counter

def skip(data, label, is_train):
    return False

def transform(data, label, is_train):
    label = one_hot(10, label)
    data = np.asarray(data, np.float32) / 255.0
    data = (data - 0.5 * np.ones((28,28))).astype('float32')
    return data, label

batch_size = 64
ds = DigitFileListDataset('svhn_600.txt', '/digits_file/', transform=transform, skip_pred=skip, is_train=True, auto_weight = False,)
source_train = CustomDataLoader(ds, batch_size=batch_size, num_threads=2)
ds2 = DigitFileListDataset('svhn_test.txt', '/digits_file/', transform=transform, skip_pred=skip, is_train=False, imsize=28)
target_test = CustomDataLoader(ds2, batch_size=1000, num_threads=2)
selectGPUs(1)
log = Logger('log/mnist-fashion', clear=True)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        
        self.fc_params = nn.Sequential(nn.Linear(50*4*4, 500), nn.ReLU(), nn.Dropout(p=0.5))
        self.classifier = nn.Linear(500, 6)
        self.__in_features = 500

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features
    
class LeNet_cls(nn.Module):
    def __init__(self):
        super(LeNet_cls, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(500, 1024),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 10)
        )
        self.softmax = nn.Softmax(dim = -1)
        self.__in_features = 500

    def forward(self, x):
        y = self.classifier(x)
        y = self.softmax(y)
        return y

    def output_num(self):
        return self.__in_features

feature_extractor_0 = LeNet().cuda()
cls = LeNet_cls().cuda()

feature_extractor_0.load_state_dict(torch.load('lenet_fashion.pkl'))
cls.load_state_dict(torch.load('lenet_fashion_cls.pkl'))

scheduler = lambda step, initial_lr : initial_lr
optimizer_feature_extractor_0 = OptimWithSheduler(optim.SGD(feature_extractor_0.parameters(), lr=0.01, weight_decay=0.0005 ,momentum=0.9), scheduler)
optimizer_cls = OptimWithSheduler(optim.SGD(cls.parameters(), lr=0.01, weight_decay=0.0005, momentum = 0.9), scheduler)

epoch=0
while epoch < 100:
    for (i, (im_source, label_source)) in enumerate(source_train.generator()):
        # =========================forward pass
        im_source = torch.unsqueeze(Variable(torch.from_numpy(im_source)).cuda(),1)
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        feature_source, _ = feature_extractor_0.forward(im_source)
        predict_prob_source = cls.forward(feature_source)
        ce = CrossEntropyLoss(label_source, predict_prob_source)
        
        with OptimizerManager([optimizer_feature_extractor_0, optimizer_cls]):
            loss = ce 
            loss.backward()
        log.step += 1

        if log.step % 10== 1:

            track_scalars(log, ['ce'], globals())

        if log.step % 100 == 0:
            clear_output()
epoch += 1


with TrainingModeManager([feature_extractor_0, cls], train=False) as mgr, Accumulator(['predict_prob','predict_index', 'label']) as accumulator:
    for (i, (im, label)) in enumerate(target_test.generator()):
        im = torch.unsqueeze(Variable(torch.from_numpy(im), volatile=True).cuda(),1)
        label = Variable(torch.from_numpy(label), volatile=True).cuda()
        fs, _ = feature_extractor_0.forward(im)
        predict_prob = cls.forward(fs)
        predict_prob, label = [variable_to_numpy(x) for x in (predict_prob, label)]
        label = np.argmax(label, axis=-1).reshape(-1, 1)
        predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
        accumulator.updateData(globals())
        if i % 10 == 0:
            print(i)

for x in accumulator.keys():
    globals()[x] = accumulator[x]

acc = float(np.sum(label.flatten() == predict_index.flatten()) )/ label.flatten().shape[0]
print(acc)
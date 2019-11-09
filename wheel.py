import pathlib2
import os
import numpy as np
import re

def setGPU(i):
    global os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(i)
    
    gpus = [x.strip() for x in (str(i)).split(',')]
    NGPU = len(gpus)
    print('gpu(s) to be used: %s'%str(gpus))
    return NGPU

def selectGPUs(n, max_load=0.1, max_memory=0.1):
    from gpuutils import getAvailable
    deviceIDs = getAvailable(order='first', limit=n, maxLoad=max_load, maxMemory=max_memory,
                             includeNan=False, excludeID=[], excludeUUID=[])
    if len(deviceIDs) < n:
        return False, None
    else:
        gpus = ','.join([str(x) for x in deviceIDs])
        setGPU(gpus)
        return True, gpus


def getHomePath():
    return str(pathlib2.Path.home())


def join_path(*a):
    return os.path.join(*a)


def ZipOfPython3(*args):
    args = [iter(x) for x in args]
    while True:
        yield [next(x) for x in args]


class AccuracyCounter:
    def __init__(self):
        self.Ncorrect = 0.0
        self.Ntotal = 0.0
    def addOntBatch(self, predict, label):
        assert predict.shape == label.shape
        correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)
        self.Ncorrect += Ncorrect
        self.Ntotal += Ntotal
        return Ncorrect / Ntotal
    
    def reportAccuracy(self):
        """
        :return: **return nan when 0 / 0**
        """
        return np.asarray(self.Ncorrect, dtype=float) / np.asarray(self.Ntotal, dtype=float)


class Accumulator(dict):
    def __init__(self, name_or_names, accumulate_fn=np.concatenate):
        super(Accumulator, self).__init__()
        self.names = [name_or_names] if isinstance(name_or_names, str) else name_or_names
        self.accumulate_fn = accumulate_fn
        for name in self.names:
            self.__setitem__(name, [])

    def updateData(self, scope):
        for name in self.names:
            self.__getitem__(name).append(scope[name])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb:
            print(exc_tb)
            return False

        for name in self.names:
            self.__setitem__(name, self.accumulate_fn(self.__getitem__(name)))

        return True

def sphere_sample(size):
    z = np.random.normal(size=size)
    z = z / (np.sqrt(np.sum(z**2, axis=1, keepdims=True)) + 1e-6)
    return z.astype(np.float32)

def sphere_interpolate(a, b, n=64):
    a = a / (np.sqrt(np.sum(a**2)) + 1e-6)
    b = b / (np.sqrt(np.sum(b**2)) + 1e-6)
    dot = np.sum(a * b)
    theta = np.arccos(dot)
    mus = [x * 1.0 / (n - 1) for x in range(n)]
    ans = np.asarray([(np.sin((1.0 - mu) * theta) * a + np.sin(mu * theta) * b) / np.sin(theta) for mu in mus], dtype=np.float32)
    return ans

def mergeImage_color(images, rows, cols=None):
    cols = rows if not cols else cols
    size = (rows, cols)
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c), dtype=images.dtype) # use images.dtype to keep the same dtype
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def mergeImage_gray(images, rows, cols=None):
    cols = rows if not cols else cols
    size = (rows, cols)
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]), dtype=images.dtype) # use images.dtype to keep the same dtype
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image
    return img

def to_gray_np(img):
    img = img.astype(np.float32)
    img = ((img - img.min()) / (img.max() - img.min()) * 1).astype(np.float32)
    return img

def to_rgb_np(img):
    img = img.astype(np.float32)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return img

def getID():
    getID.x += 1
    return getID.x
getID.x = 0

def clear_output():
    def clear():
        return
    try:
        from IPython.display import clear_output as clear
    except ImportError as e:
        pass

    import os

    def cls():
        os.system('cls' if os.name == 'nt' else 'clear')

    clear()
    cls()


class Nonsense:
    def __getattr__(self, item):
        if item not in self.__dict__:
            self.__dict__[item] = Nonsense()
        return self.__dict__[item]

    def __call__(self, *args, **kwargs):
        return Nonsense()

    def __str__(self):
        return "Nonsense object!"

    def __repr__(self):
        return "Nonsense object!"

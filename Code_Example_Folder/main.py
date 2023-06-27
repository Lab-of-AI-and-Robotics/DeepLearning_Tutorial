import torch.nn as nn
from model import SegNet
from PIL import Image
import torchvision
import tqdm
from utils import *
import cv2
from torchvision.utils import save_image
import torch.nn.functional as F

device = 'cuda:0'


root_path = '/content/gdrive/MyDrive/2022_2/Comvi/ass3/'
class Dataset(object):
    def __init__(self, img_path, label_path, method='train'):
        self.img_path = img_path
        self.label_path = label_path
        self.train_dataset = []
        self.test_dataset = []
        self.mode = method == 'train'
        self.preprocess()
        if self.mode:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(
                len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])):
            img_path = os.path.join(self.img_path, str(i) + '.jpg')
            label_path = os.path.join(self.label_path, str(i) + '.png')
            print(img_path, label_path)
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]
        image = Image.open(img_path)
        label = Image.open(label_path)
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((512, 512))])
        return transform(image), transform(label), img_path.split("/")[-1]

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class Tester(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.model = self.build_model()
        # Load of pretrained_weight file
        # weight_PATH = '/content/gdrive/MyDrive/2022_2/Comvi/ass3/final_model.pth'
        weight_PATH = '/content/gdrive/MyDrive/2022_2/Comvi/ass3/model__20_.pth'
        self.model.load_state_dict(torch.load(weight_PATH))
        dataset = Dataset(img_path=root_path + "data/test_img/", label_path=root_path + "data/test_label/", method='test')
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      drop_last=False)
        print("Testing...")

    def test(self):
        make_folder("test_mask", '')
        make_folder("test_color_mask", '')
        self.model.eval()
        for i, data in enumerate(self.dataloader):
            imgs = data[0].cuda()
            labels_predict = self.model(imgs)
            labels_predict_plain = generate_label_plain(labels_predict, 512)
            labels_predict_color = generate_label(labels_predict, 512)
            # if i == 1:
            #   print('labels_predict_plain', labels_predict_plain)
            #   print('labels_predict_color', labels_predict_color)
            #   break
            batch_size = labels_predict.size()[0]
            for k in range(batch_size):
                cv2.imwrite(os.path.join("test_mask", data[2][k]), labels_predict_plain[k])
                save_image(labels_predict_color[k], os.path.join("test_color_mask", data[2][k]))
        
        print('Test done!')

    def build_model(self):
        model = SegNet(3).cuda()
        return model


class Trainer(object):
    def __init__(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.model = self.build_model()
        # I set loss function cross entropy and Adam optimizer(Already define origin_main.py)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        dataset = Dataset(img_path=root_path + "data/train_img", label_path=root_path + "data/train_label/", method='train')
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      drop_last=False)


        
        # Load of pretrained_weight file
        weight_PATH = '/content/gdrive/MyDrive/2022_2/Comvi/ass3/final_model.pth'
        self.model.load_state_dict(torch.load(weight_PATH))
    
    
    
    def train(self):
        cost = self.criterion
        optimizer = self.optimizer
        for epoch in tqdm.tqdm(range(self.epochs + 1)):
            for batch in self.dataloader:
                # load input and target image
                input_img =  batch[0]
                target =  batch[1]
                
                # target img was nomailized, so multiply 255
                target = target * 255.0
                
                # using segnet, make predicted img, and target image chage dimension and tpye to long.
                predicted = self.model(input_img.cuda())
                target = target.view(-1,512,512)
                target = target.long()
                
                # get loss and backward, weight update
                optimizer.zero_grad()
                loss = cost(predicted, target.cuda())
                print("loss is", loss)
                loss.backward()
                optimizer.step()
                
            if epoch % 1 == 0:
                torch.save(self.model.state_dict(), "_".join(['/content/gdrive/MyDrive/2022_2/Comvi/ass3/save_weight/model_', str(epoch), '.pth']))

                
                
        print('Finish training.')
                
                


    def build_model(self):
        model = SegNet(3).cuda()
        return model


if __name__ == '__main__':
    epochs = 20
    lr = 0.001 # i change 0.01 to 0.001 to learning stability
    batch_size = 32
    # trainer = Trainer(epochs, batch_size, lr)
    # trainer.train()
    tester = Tester(32)
    tester.test()
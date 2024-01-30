import torch
from numpy import mean 
from torch.utils.data import random_split
from torchvision import datasets as ds 
from torchvision import transforms as tfs
import glob
from numpy import array
from PIL import Image
class Trainer():
    """
    Defines how training of Autoenconder is taking place
    """
    def __init__(self,modelD,modelG, device, optD,optG, lf):
        """_Initializes trainer_

        Args:
            model (_torch.nn.Module_): _description_
            device (_torch.device_): _cpu or cuda_
            opt (_torch.optim_): _choosen optimizer function_
            lf (_torch.nn_): _choosen  loss function_
        """
        self.dev = device
        self.mod = modelG
        self.disc= modelD
        self.optD = optD
        self.optG = optG
        self.lf = lf
        self.rl= 1.
        self.fl= 0.
       
      
    def train(self, train_loader):
        """_Train the model for n elements in the train loader_

        Args:
            train_loader : _Train loader object with elements for the model to train_

        Returns:
            _double_: _Returns mean loss value of the batch_
        """
        
        trainR_loss=[]
        trainF_loss =[]
        D_x=0
        D_G_z1=0
        D_G_z2=0
        for (image, i) in train_loader:
            
            self.disc.zero_grad()
            # Train real
            real_cpu = image.to(self.dev)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), self.rl, dtype=torch.float, device=self.dev)
        
            output = self.disc(real_cpu).view(-1)
            lossR=self.lf(output,label)
            lossR.backward()
            D_x = output.mean().item()
            trainR_loss.append(lossR.detach().cpu().numpy())
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, 50, 1, 1, device=self.dev)
            # Generate fake image batch with G
            fake = self.mod(noise)
            label.fill_(self.fl)
            # Classify all fake batch with D
            output = self.disc(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            lossF = self.lf(output, label)
            lossF.backward()
            trainF_loss.append(lossF.detach().cpu().numpy())

            loss = lossR + lossF
            D_G_z1 = output.mean().item()
            #Update
            self.optD.step()
            ### Train G  
            self.mod.zero_grad() 
            label.fill_(self.rl)
            output = self.disc(fake).view(-1)
            lossG = self.lf(output,label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            self.optG.step()
        return mean(trainR_loss),mean(trainF_loss), D_x, D_G_z1,D_G_z2
    

    def get_model(self):
        """_Gets current model trained/evaluated_

        Returns:
            _torch.nn.Module_: _trained model_
        """
        return self.mod
    def get_D(self):
        return self.disc
    
    def get_optD(self):
        return self.optD
    
    def get_optG(self):
        return self.optG

def MNIST_loader(ds_folder,batch_size=256, color=False):
    train_dataset = ds.MNIST(ds_folder,train=True,download=True)
    test_dataset = ds.MNIST(ds_folder,train=False,download=True)
    # transform to grayscale
    train_tf = tfs.Compose([tfs.ToTensor(),tfs.Grayscale(),])
    test_tf=tfs.Compose([tfs.ToTensor(),tfs.Grayscale(),])
    
    m=len(train_dataset)
    
    train_dataset.transform=train_tf
    test_dataset.transform = test_tf
    train_data, val_data = random_split( train_dataset,  [int(m-m*0.2), int(m*0.2)])
    # loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    return train_loader, val_loader,test_loader,test_dataset


def custom_loader(train_ds, h=64,w=64,color = True, batch = 10):
   
    # transform to grayscale
    if color is False:
        dataset=ds.ImageFolder(root=train_ds,transform= tfs.Compose([tfs.Resize((h, w)),tfs.ToTensor(),tfs.Grayscale(),
                                tfs.Normalize((0.5),(0.5)),
                            ]))
        
    else:
        dataset_og = ds.ImageFolder(root = train_ds,
                            transform=tfs.Compose([tfs.Resize((h, w)),tfs.ToTensor(),
                                tfs.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                            ]))
        # Augment the dataset with mirrored images
        mirror_dataset = ds.ImageFolder(root = train_ds,
                            transform=tfs.Compose([tfs.Resize((h, w)),
                                    tfs.RandomHorizontalFlip(p=1.0),
                                    tfs.ToTensor(),
                                tfs.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                            ]))
        sharp_dataset = ds.ImageFolder(root = train_ds,
                            transform=tfs.Compose([tfs.Resize((h, w)),
                                tfs.RandomAdjustSharpness(sharpness_factor=2),
                                tfs.ToTensor(),
                                tfs.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                            ]))
        autocontrast_dataset=ds.ImageFolder(root = train_ds,
                            transform=tfs.Compose([tfs.Resize((h, w)),
                                tfs.RandomAutocontrast(),
                                tfs.ToTensor(),
                                tfs.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                            ]))
    dataset = torch.utils.data.ConcatDataset([dataset_og,mirror_dataset,autocontrast_dataset])     
    # loaders
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch,
                                         shuffle=True,num_workers=2)
    return train_loader

 
class SparrowDataset(torch.utils.data.Dataset):
    """_Maps Dataset for FZI Project_"""

    def __init__(self, root_dir,color = False,transform = None):
        file_list = glob.glob(root_dir + "/*")
        self.data = []
        labels=[]
        self.targets=[]
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            label = 0 if class_name.lower()=='train' else 1
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
            self.class_map = { "train":0, "test":1}

        self.transform = transform
        #self.samples = samples
        self.targets=[s[0] for s in self.targets]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        class_id = self.class_map[class_name]      
        class_id = torch.tensor([class_id])
        if self.transform is not None:
            img_tensor = self.transform(img)

        #if self.targets is not None:

        return img_tensor, class_id
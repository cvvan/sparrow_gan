import os
import torch 
from models.gan import Generator as G, Discriminator as D
from matplotlib import pyplot as plt
import numpy as np
import random
import hydra 
import utils.Trainer as uTrainer
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter as OutputTo

# Get current working directory - work only in 
# the current top directory!! ../rl-autoencoder/
path = os.getcwd()
path_to_cwd = ""

manualSeed = 814
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) 
# check if cuda available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

# hydra config
@hydra.main(config_path='conf', config_name='config')
def __main__(cfg: DictConfig)->None:
    run = 2
    # Define directories
    outimgs =path + path_to_cwd+str(cfg.data.out_imgs)
    outtb = path + path_to_cwd + str(cfg.data.out_tb)
    ds_fd= path + path_to_cwd + str(cfg.datasets.folder)
    try:
        os.makedirs(outimgs)
        os.makedirs(outtb)
    except FileExistsError:
        pass
    # Creating batch of images
    train_ds = uTrainer.SparrowDataset(ds_fd )
    train_loader = uTrainer.custom_loader(ds_fd,              
                                                    cfg.model.image.size,cfg.model.image.size,
                                                    cfg.model.image.color,cfg.model.image.batch)
    print("Train Loader finished")

    # Configuring SummaryWriter
    writer = OutputTo(outtb,comment="First try")
    
    
    ######
    # Create the generator
    netG = G(3).to(device)
    if torch.cuda.is_available():
        netG.cuda()

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = D(3).to(device)

    if torch.cuda.is_available():
        netD.cuda()

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    #######   
    # define loss function
    # ist MSE es gut für Bilder?
    loss_function = torch.nn.BCELoss()
    #Noise
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, 50, 1, 1, device=device) # 50 size og gen input
    
    # Using an Adam Optimizer
    optimizerD = torch.optim.Adam(netD.parameters(),
                                lr = cfg.training.lr,
                                betas=(cfg.training.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(),
                                lr = cfg.training.lr,
                                betas=(cfg.training.beta1, 0.999))
    
    epochs = cfg.training.epochs
    trainR_loss = []
    trainF_loss = []
    img_list = []
    #initialize trainer
    trainer = uTrainer.Trainer(netD,netG,device,optimizerD,optimizerG,loss_function)
    for epoch in range(epochs):
        ##############
        ### TRAIN  ###
        ##############
        lossR,lossF,D_x,D_G_z1,D_G_z2= trainer.train(train_loader=train_loader)
        trainR_loss.append(lossR)
        trainF_loss.append(lossF)
        # Adding train loss to tensorboard per epoch
        writer.add_scalar('LossReal/train', np.mean(trainR_loss), epoch)
        writer.add_scalar('LossFake/train', np.mean(trainF_loss), epoch)
        writer.add_scalar('D_x/train', D_x, epoch)
        writer.add_scalar('D_G_z1/train', D_G_z1, epoch)  
        writer.add_scalar('D_G_z2/train', D_G_z2, epoch)
        #print(train_loss)
      
        ################
        ###  VALIDATE ###
        #################
        #val_loss=trainer.validate(val_loader=val_loader)
        # Adding validation loss to tensorboard per epoch
        #writer.add_scalar('Loss/validation', val_loss, epoch)

        #@DEPRECATED:
        print('\n EPOCH {}/{} \t real train loss {} \t fake train loss {} '.
              format(epoch + 1, epochs,np.mean(trainR_loss),np.mean(trainF_loss)))
        # Get model
        model = trainer.get_model()
        ##################
        ## VISUALIZATION #
        ##################
        ##https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
        # Part 4
        if epoch % 5 == 0:
            plt.figure(figsize=(10,10)),
            classes=1
            #targets = test_ds.targets.numpy()
            #t_idx = {i:np.where(targets==i)[0][0] for i in range(classes)}
            for i in range(classes):
                model.eval()
                with torch.no_grad():
                    reconstructed = model(fixed_noise).detach().cpu()
                img_list.append(make_grid(reconstructed,padding=2, normalize=True))
            # Plot the fake images from the last epoch
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list[-1],(1,2,0)))      
            plt.savefig(outimgs + "rec"+str(epoch) )
            plt.close()
    writer.flush()
    state = {
        'epoch': epoch,
        'stateG_dict': model.state_dict(),
        'stateD_dict':trainer.get_D.state_dict(),
        'optimizerD': trainer.get_optD.state_dict(),
        'optimizerG':trainer.get_optG.state_dict(),
    }
    torch.save(state, outimgs +"state.pyc")
if __name__ == '__main__':
    __main__()

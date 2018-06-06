import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")

#Data loading
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)


#Defining the neural networks
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(64, 128)
        self.layer2 = nn.Linear(128, 512)
        self.layer3 = nn.Linear(512, 28*28)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.tanh(self.layer3(x))
        return x
    
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(28*28,256)
        self.layer3 = nn.Linear(256, 1)


    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.sigmoid(self.layer3(x))
        return x


discriminator = Discriminator()
generator = Generator()


criterion = nn.BCELoss()
optimizerGenerator = optim.SGD(generator.parameters(), lr=0.001)
optimizerDiscriminator = optim.SGD(discriminator.parameters(), lr=0.05)

#Training the GAN
no_epochs = 250
m = 16 #mini-batch size

#dataset filtering
data = torch.zeros(5923,28,28)
i = 0
label = 0
for k in range(len(trainset.train_labels)):
    if (trainset.train_labels[k].item() == label):
        data[i] = trainset.train_data[k]
        data[i] = 2*data[i]/255-1 #normalization
        i += 1


data = data[:5920]

data = data.reshape(5920,28*28)

loss_gen_history=[]
loss_discr_history=[]

y = 0

for epoch in range(no_epochs):
    
    running_loss_gen = 0.0
    running_loss_discr = 0.0
        
    for batch in range(int(5920/m)):

        x = []
        k = 1 #number of steps of discriminator training per generator training step
        
        for i in range(k):
            #Discriminator training
            optimizerDiscriminator.zero_grad()
        
            noise = torch.randn(m, 64, device=device, dtype=dtype)
            x_gen = generator(noise)

            x_data = data[m*batch:m*(batch+1)]
            
            assert x_data.shape == x_gen.shape

            x = torch.zeros(2*m,28*28, dtype=dtype)
            for example in range(m):
                x[example] = x_gen[example]
            for example in range(m,2*m):
                x[example] = x_data[example-m]

            
            y = discriminator(x)
            
            y = y.reshape(2*m)
        
    
            target = torch.ones(2*m, dtype=torch.float)
            target[:m] = torch.zeros(m, dtype=torch.float)
            
            loss = criterion(y, target)
        
            loss.backward()
            optimizerDiscriminator.step()
            
            running_loss_discr += loss.item()
        
        #Generator training
        optimizerGenerator.zero_grad()
        
        noise = torch.randn(m, 64, device=device, dtype=dtype)
        x_gen = generator(noise)
        
        y_gen = discriminator(x_gen)
        y_gen = y_gen.reshape(m)
        
        #Labels flipping (generator training)
        target = torch.ones(m, dtype=torch.float)
            
        loss = criterion(y_gen, target)
        
        loss.backward()
        optimizerGenerator.step()
        
        running_loss_gen += loss.item()
        

        if batch % 50 == 49:    # print every 50 batch
            
            loss_gen_history.append(running_loss_gen/50)
            loss_discr_history.append(running_loss_discr/50)
            
            print('[%d, %5d] Generator loss: %.7f' %(epoch + 1, batch + 1, running_loss_gen / 50))
            print('[%d, %5d] Discriminator loss: %.7f' %(epoch + 1, batch + 1, running_loss_discr / 50))
            running_loss_gen = 0.0
            running_loss_discr = 0.0
            
            img = x[0]
            img = img.detach().numpy()
            img = img.reshape((28,28))
            plt.imshow(img,cmap='Greys')
            plt.show()
            plt.plot([k for k in range(len(loss_gen_history))],loss_gen_history, loss_discr_history)
            plt.show()
            

#Testing
no_to_generate = 10

#Using no_grad() because we don't want to track gradients during testing
with torch.no_grad():
    
    
    noise = torch.randn(200, 64, device=device, dtype=dtype)
    x_gen = generator(noise)
    print("variance of generator = " + str(x_gen.std(0).mean().item()))
    
    for k in range(no_to_generate):
        noise = torch.randn(1, 64, device=device, dtype=dtype)
        x_gen = generator(noise)
        
        img = x_gen.detach().numpy()
        img = img.reshape((28,28))
        plt.imshow(img, cmap='Greys')
        plt.savefig('generated image no '+str(k)+' .png')
    
    plt.clf()
    plt.figure(figsize=(10,4))
    plt.plot([k for k in range(len(loss_gen_history))],loss_gen_history, loss_discr_history)
    plt.savefig('loss.png')
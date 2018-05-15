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
    #3-layer perceptron
    #input = vector of dimension 28*28 (noise)
    #output = 28*28*1 image
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(14, 14)
        self.layer2 = nn.Linear(14, 28)
        self.layer3 = nn.Linear(28, 28*28)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.tanh(self.layer3(x))
        x = x.view(-1,28,28)
        return x
    
class Discriminator(nn.Module):
    #ConvNet : 2 layers Conv2d / 3 dense layers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.fc1 = nn.Linear(4 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


discriminator = Discriminator()
generator = Generator()


criterion = nn.CrossEntropyLoss()
optimizerGenerator = optim.SGD(generator.parameters(), lr=0.02)
optimizerDiscriminator = optim.SGD(discriminator.parameters(), lr=0.005)

#Training the GAN
no_epochs = 30
m = 10 #mini-batch size

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

loss_gen_history=[]
loss_discr_history=[]

y = 0

for epoch in range(no_epochs):
    
    running_loss_gen = 0.0
    running_loss_discr = 0.0
        
    for batch in range(int(5920/m)):

        x = []
        k = 1
        
        for i in range(k):
            #Discriminator training
            optimizerDiscriminator.zero_grad()
        
            noise = torch.rand(m, 14, device=device, dtype=dtype)
            x_gen = generator(noise)
            
            x_gen= x_gen.view(m,1,28,28)
            
            x_data = data[m*batch:m*(batch+1)].view(m,1,28,28)
            
            assert x_data.shape == x_gen.shape
            
            x = torch.zeros(2*m,1, 28, 28, dtype=dtype)
            for example in range(m):
                x[example] = x_gen[example]
            for example in range(m,2*m):
                x[example] = x_data[example-m]

            
            y = discriminator(x)

    
            target = torch.ones(2*m, dtype=torch.long)
            target[:m] = torch.zeros(m, dtype=torch.long)
            
            loss = criterion(y, target)
        
            loss.backward()
            optimizerDiscriminator.step()
            
            running_loss_discr += loss.item()
        
        #Generator training
        optimizerGenerator.zero_grad()
        
        noise = torch.rand(m, 14, device=device, dtype=dtype)
        x_gen = generator(noise)
        x_gen= x_gen.view(m,1,28,28)
        
        y_gen = discriminator(x_gen)
        
        target = torch.ones(m, dtype=torch.long)
            
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
            plt.imshow(img)
            plt.show()
            plt.plot([k for k in range(len(loss_gen_history))],loss_gen_history, loss_discr_history)
            plt.show()


#Testing
no_to_generate = 10

#Using no_grad() because we don't want to track gradients during testing
with torch.no_grad():
    for k in range(no_to_generate):
        noise = torch.rand(1, 14, device=device, dtype=dtype)
        x_gen = generator(noise)
        
        img = x_gen.detach().numpy()
        img = img.reshape((28,28))
        plt.imshow(img)
        plt.savefig('generated image no '+str(k)+' .png')

    plt.plot([k for k in range(len(loss_gen_history))],loss_gen_history, loss_discr_history)
    plt.savefig('loss.png')
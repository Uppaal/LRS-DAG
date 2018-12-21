import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets

import pickle
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.E0 = nn.Linear(1024, 1024)
        
        self.N1 = nn.Sequential(nn.Linear(1024, 512),
                               nn.Linear(512, 256))
        
        self.E1 = nn.Sequential(nn.Linear(256, 64),
                               nn.Linear(64, 256))
        
        self.N2 = nn.Sequential(nn.Linear(256, 64),
                               nn.Linear(64, 10))
        
                
    def forward(self, x, use_E_layers=False, get_embedding=False):        
        out = x.view(x.shape[0],-1)
        out = self.N1(out)
        
        if use_E_layers == False and get_embedding == True:
            return F.softmax(out)
        
        # Encoder block
        if use_E_layers == True:
            out = self.E1(out)
            if get_embedding == True:
                return F.softmax(out)
        
        out = self.N2(out)
        out = F.softmax(out)
        return out


class Trainer(Net):
    def __init__(self, model_name, learning_rate_target, weight_decay_target, num_epochs):
        super(Trainer, self).__init__()
        self.learning_rate_source = 0.0001
        self.weight_decay_source = 1e-6
        self.batch_size_source = 128
        self.num_epochs_source = 10
        
        self.learning_rate_target = learning_rate_target
        self.weight_decay_target = weight_decay_target
        self.batch_size_target = 128
        self.num_epochs_target = num_epochs
        
        self.model_path = model_name
        self.device = ['cuda' if torch.cuda.is_available() else 'cpu']
        self.model = Net().to(self.device[0])
        
###########################################################################################################################        
        
    def fit_source(self, train_set):
        """Fit the model to the source domain data"""
        
        print("\nSource Domain:\n")
        self.model.train()
        
        # Create data loader for source data
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.batch_size_source, shuffle=True)
    
        # Use cross entropy Loss to train the model on the source data
        loss_fn = nn.CrossEntropyLoss()
        
        # Use the Adam algorithm for gradient descent
        optimizer = optim.Adam(self.model.parameters(), weight_decay=self.weight_decay_source, lr=self.learning_rate_source)
    
        losses = []
        
        # Train over Epochs
        for epoch in range(self.num_epochs_source):
            loss_val = []
        
            for x, y in train_loader:
                x = x.to(self.device[0])
                y = y.to(self.device[0])
                
                # Set gradients for new epoch to zero
                optimizer.zero_grad()
        
                # Calculate the loss
                output = self.model(x)
                loss = loss_fn(output, y)
                loss_val.append(float(loss))
        
                # Calculate gradients for backpropagation
                loss.backward()
            
                # Update model parameters
                optimizer.step()
            
            # Find loss for epoch
            losses.append(np.average(loss_val))
            print("Epoch ", epoch+1, ": loss = ", np.average(loss_val))
                
        # Save model 
        self.save_model()

###########################################################################################################################                
                                                          
    def fit_target(self, source_train_set, target_train_set, method=1, use_E_layers=True, variational=True):
        print("\nTarget Domain:\n")
        self.model.train()

        # Load pretrained source weights
        self.load_model()

        # Create data loaders for source  and target data
        source_train_loader = torch.utils.data.DataLoader(dataset=source_train_set, batch_size=len(source_train_set))
        target_train_loader = torch.utils.data.DataLoader(dataset=target_train_set, batch_size=self.batch_size_target, shuffle=True)

        # Freeze the weights of non trainable layers
        if method=='finetune-only':
            use_E_layers == False
            self.freeze_layers([0,1,2])
        else:
            self.freeze_layers([0,1,3])

        # Define loss, based on what method is being used
        if method == 1 or method=='finetune-only':
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_class = nn.CrossEntropyLoss()
        if method == 2:
            loss_fn = nn.MSELoss()
        if method == 3 or method == 5:
            loss_fn = nn.KLDivLoss()
        if method == 4:
            loss_fn = nn.MSELoss()

        # Pass only weights of Encoder layers to the optimizer
        optimizer = optim.Adam([param for param in self.model.parameters() if param.requires_grad == True], weight_decay=self.weight_decay_target, lr=self.learning_rate_target)

        # Get embeddings from source domain
        source_embedding, mu_s, sigma_s = self.get_embedding_distribution(source_train_loader, use_E_layers=False, use_loader=True)

        # Train over Epochs
        losses = []
        for epoch in range(self.num_epochs_target):

            loss_val = []

            for x, y in target_train_loader:
                x = x.to(self.device[0])
                y = y.to(self.device[0])

                # Set gradients for new epoch to zero
                optimizer.zero_grad()                

                # Get the output of the model
                output = self.model(x, use_E_layers=use_E_layers)

                if method == 1 or method == 'finetune-only':
                    loss = loss_fn(output, y)
                else:                
                    target_embedding, mu_t, sigma_t = self.get_embedding_distribution(x, use_E_layers=True)

                    # Sample source embeddings for the particular batch
                    if variational == True:
                        # Estimate source distribution and sample from that distribution to get source embedding
                        source_embedding_batch = np.random.multivariate_normal(mu_s, sigma_s, len(y))                    
                        source_embedding_batch = torch.from_numpy(source_embedding_batch).float().to(self.device[0])
                    else:
                        # Randomly sample data points from source embedding
                        random_idx = np.random.randint(low=0, high=len(source_embedding), size=len(y))
                        source_embedding_batch = source_embedding[random_idx]        

                if method == 2 or method == 3:
                    loss = loss_fn(target_embedding, source_embedding_batch) + loss_class(output, y)
                if method == 4:
                    loss = loss_fn(mu_s, mu_t) + loss_fn(sigma_s, sigma_t) + loss_class(output, y)
                if method == 5:
                    loss = loss_fn(source_embedding_batch, target_embedding) + loss_class(output, y)
                if method == 6:
                    loss = self.CORAL_loss(sigma_s, sigma_t)

                # Calculate gradients for backpropagation
                loss_val.append(float(loss))
                loss.backward()            

                # Update model parameters
                optimizer.step()

            # Find loss for epoch
            losses.append(np.average(loss_val))
            print("Epoch ", epoch+1, ": loss = ", np.average(loss_val))

    
###########################################################################################################################                    
    
    def get_performance(self, source_test_set, target_test_set):
        print("Without E on source: ", self.test(source_test_set, use_E_layers=False), "%")
        print("Without E on target: ", self.test(target_test_set, use_E_layers=False), "%")
        print("With E on source: ", self.test(source_test_set, use_E_layers=True), "%")
        print("With E on target: ", self.test(target_test_set, use_E_layers=True), "%")
    
    def test(self, test_set, use_E_layers):       
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128)
        
        correct = 0
        with torch.no_grad():
            for X, Y in test_loader:
                X = X.to(self.device[0])
                Y = Y.to(self.device[0])
                Y_hat = self.model(X, use_E_layers=use_E_layers)
                Y_hat = Y_hat.data.max(1, keepdim=True)[1]
                correct += Y_hat.eq(Y.data.view_as(Y_hat)).sum()

        return float(100. * float(correct) / float(len(test_loader.dataset)))
    
###########################################################################################################################                
      
    def CORAL_loss(self, sigma_s, sigma_t):
        loss = torch.mean(torch.mul((sigma_s - sigma_t), (sigma_s - sigma_t)))
        return loss
    
    def get_embedding_distribution(self, data, use_E_layers, use_loader=False):
        # If getting embeddings from N1 on source, 'data' will be a data loader.
        if use_loader == True:
            for (x, y) in data:
                x = x.to(self.device[0])
                embedding = self.model(x, use_E_layers=use_E_layers, get_embedding=True)    # N x 256

        # If getting embeddings from E on target, data will be passed in batches, from the loader.
        elif use_loader == False:
            embedding = self.model(data, use_E_layers=use_E_layers, get_embedding=True)    # N x 256

        mu_hat = torch.mean(embedding, 0)
        mu_hat = mu_hat.float().to(self.device[0])                    # 256

        sigma_hat = torch.mean(embedding, 0, keepdim=True) - embedding
        sigma_hat = sigma_hat.t() @ sigma_hat                        # 256 x 256
        sigma_hat = sigma_hat.float().to(self.device[0])

        return embedding, mu_hat , sigma_hat
    
###########################################################################################################################                    
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("Model Saved: ", self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print("Model Loaded: ", self.model_path)

    def freeze_layers(self, freeze_indices):    
        i = 0
        for child in self.model.children():
            if i in freeze_indices:
                for param in child.parameters():
                    param.requires_grad = False
            i  = i + 1

    def print_weights(self, layer_indices, epoch):    
        i = 0
        for child in self.model.children():
            if i in layer_indices:
                print ("\nEpoch ", epoch, ": ")
                j = 0
                for param in child.parameters(): 
                    if j % 2 != 0:
                        print (param[:1][:5])
                    j = j+1
            i = i + 1
            
###########################################################################################################################                

    def plot_embeddings(self, X, num_images=10):
        i = 0
        images = []
        for i in range(num_images):
            x = X[i]
            im_width = int(np.sqrt(x.shape[0]))
            x = x.cpu().detach().numpy().reshape([im_width, im_width])
            images.append(x)

        plt.figure(figsize=(10,10))
        for i in range(len(images)):
            rows, cols = int(num_images/5), 5
            plt.subplot(rows, cols, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
        plt.show()

    def plot_tsne(self, X, Y=None, use_loader=False, num_points=500, title=None, filename=None, show=False):
        from sklearn.manifold import TSNE
        if use_loader == True:
            loader = torch.utils.data.DataLoader(dataset=X, batch_size=num_points)
            for (x,y) in loader:
                X = x.cpu().detach().numpy().reshape([x.shape[0], -1])
                Y = y.cpu().detach().numpy()
                break
        else:
            X = X.cpu().detach().numpy().reshape([X.shape[0], -1])
            Y = Y.cpu().detach().numpy()
            X = X[:num_points]
            Y = Y[:num_points]

        X_plt = TSNE(n_components=2, random_state=7, learning_rate=100).fit_transform(X)   # N x 2

        plt.figure(figsize=(6, 5))
        target_ids = range(10)   # Number of classes
        colors = ['dimgrey', 'salmon', 'yellow', 'lawngreen', 'green', 'cyan', 'mediumblue', 'blueviolet', 'crimson', 'fuchsia']

        for i, c, label in zip(target_ids, colors, target_ids):
            plt.scatter(X_plt[Y == i, 0], X_plt[Y == i, 1], c=c, label=label)

        plt.title(title)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        if show:
            plt.show()
        else:
            plt.savefig(filename)

###########################################################################################################################                



def load_data(source='MNIST', target='SVHN'):
    
    image_size = (32, 32)
    transform = torchvision.transforms.Compose([torchvision.transforms.Scale(image_size), torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    if source == 'MNIST':
        source_train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
        source_test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    if target == 'SVHN':
        target_train_set_full = datasets.SVHN('./data', split='train', download=True, transform=transform)
        keep_target_fraction = int(len(target_train_set_full) * 0.1)
        target_train_set = torch.utils.data.random_split(target_train_set_full, lengths=[keep_target_fraction, len(target_train_set_full)-keep_target_fraction])
        target_train_set = target_train_set[0]
        target_test_set = datasets.SVHN('./data', split='test', download=True, transform=transform)

    if target == 'MNIST':
        target_train_set = source_train_set
        target_test_set = source_test_set
        
    if target == 'Syn-MNIST':
        transform = torchvision.transforms.Compose([torchvision.transforms.Scale(image_size), 
                                            torchvision.transforms.Grayscale(num_output_channels=1),
                                            torchvision.transforms.ColorJitter(brightness=100, contrast=100, saturation=50),
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            #torchvision.transforms.RandomAffine(10, shear=15),
                                            torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        target_train_set_full = datasets.MNIST('./data', train=True, download=True, transform=transform)
        keep_target_fraction = int(len(target_train_set_full) * 0.1)
        target_train_set = torch.utils.data.random_split(target_train_set_full, lengths=[keep_target_fraction, len(target_train_set_full)-keep_target_fraction])
        target_train_set = target_train_set[0]
        target_test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

    return source_train_set, source_test_set, target_train_set, target_test_set



def main():
    # Load datasets
    source_train_set, source_test_set, target_train_set, target_test_set = load_data()
    
    # Initialize the model class
    model = Trainer("source_trained.pkl", learning_rate_target=0.001, weight_decay_target=0, num_epochs=4)
        
    # Train the model on the Source Domain, save the model
    model.fit_source(source_train_set)
    print("Before training E (training N1 and N2 on source): ")
    model.get_performance(source_test_set, target_test_set)
    
    # Train the Encoder on the Target Domain, and test performance of the final model on source and target
    model.fit_target(source_train_set, target_train_set, method=6)
    
    print("After training E (on target): ")
    model.get_performance(source_test_set, target_test_set)


if __name__ == '__main__':
    main()


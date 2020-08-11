import torch



class Trainer:
    def __init__(
        self, model,
        dataloader = None,
        grad_clip_value = 2.,
        learning_rate = 1e-3,
        device = 'cpu'
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        # gradient cliping
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -grad_clip_value, grad_clip_value))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def compute_loss(y_hat, y):
        '''Compute RMSE loss'''
        return torch.sqrt(torch.mean((y_hat-y)**2))


    def training_step(self, dataloader = None, writer = None):
        '''Produce a training step over dataset and return average loss'''
        if not (self.dataloader or dataloader):
            raise ValueError('You should provide dataloader')

        train_dataloader = dataloader if dataloader else self.dataloader

        avg_loss = 0.
        for ind, (x,) in enumerate(train_dataloader):
            x = x.to(self.device)
            target = x.flip(1) # reverse along time

            recon = self.model(x)

            loss = Trainer.compute_loss(recon, target)

            #print(f'\r{ind+1}/{len(train_dataloader)} | {loss.item()}', end='')
            avg_loss += loss.item()
            if writer:
                writer.add_scalar('RMSE', loss.item(), ind)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss /= len(train_dataloader)

        return avg_loss

    def train(self, count_epochs, dataloader = None, writer = None):
        '''Iterate over epochs and return losses history.
        Writer is instance of torch.utils.tensorboard writer'''
        losses_history = []
        for epoch in range(count_epochs):
            loss_on_epoch = self.training_step(dataloader, writer)
            losses_history.append(loss_on_epoch)

            if writer:
                writer.add_scalar('Average loss on epoch', loss_on_epoch, epoch)
            print(f'{epoch + 1} / {count_epochs} | Loss: {loss_on_epoch}')

        return losses_history


    def extract_features(self, dataloader):
        self.model = self.model.eval()

        with torch.no_grad():
            features = [self.model.get_features(x.to(self.device)) for (x,) in dataloader]
            features = torch.cat(features, 0)

        return features

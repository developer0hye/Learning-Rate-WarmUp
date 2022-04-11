# PyTorch-Learning-Rate-WarmUp

https://github.com/developer0hye/Torch-Warmup

![image](https://user-images.githubusercontent.com/35001605/125312714-7d6b6f80-e36f-11eb-9638-67e77ae6b94c.png)

## Implementation

```python
import torch
import matplotlib.pyplot as plt

class LearningRateWarmUP(object):
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.step(1)

    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr*float(cur_iteration)/float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iteration):
        if cur_iteration <= self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            self.after_scheduler.step(cur_iteration-self.warmup_iteration)
    
    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict)

if __name__ == '__main__':
    v = torch.zeros(10)
    lr = 1e-2
    total_iter = 100
    warmup_iter = 10

    optim = torch.optim.SGD([v], lr=lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_iter-warmup_iter)
    scheduler = LearningRateWarmUP(optimizer=optim,
                                   warmup_iteration=warmup_iter,
                                   target_lr=lr,
                                   after_scheduler=scheduler_cosine)

    x_iter = [0]
    y_lr = [0.]

    for iter in range(1, total_iter+1):
        print("iter: ", iter, " ,lr: ", optim.param_groups[0]['lr'])

        optim.zero_grad()
        optim.step()

        scheduler.step(iter)
        
        x_iter.append(iter)
        y_lr.append(optim.param_groups[0]['lr'])
    
    plt.plot(x_iter, y_lr, 'b')
    plt.legend(['learning rate'])
    plt.xlabel('iteration')
    plt.ylabel('learning rate')
    plt.show()

```

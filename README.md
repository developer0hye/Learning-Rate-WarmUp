# PyTorch-LerarningRate-WarmUp

```python
class LearningRateWarmUP(object):
    def __init__(self, optimizer, target_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.target_iteration = target_iteration
        self.target_lr = target_lr
        self.num_iterations = 0
        self.after_scheduler = after_scheduler

    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr*float(cur_iteration)/float(self.target_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iteration):
        if cur_iteration <= self.target_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            self.after_scheduler.step(cur_iteration)

v = torch.zeros(10)
optim = torch.optim.SGD([v], lr=lr)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.iters)
scheduler = LearningRateWarmUP(optimizer=optim, target_iteration=10, target_lr=0.01, after_scheduler=scheduler_cosine)
for epoch in range(1, 20):
    optim.zero_grad()
    optim.step()

    scheduler.step(epoch)
    print(epoch, optim.param_groups[0]['lr'])
```

## Results

```
1 0.001
2 0.002
3 0.003
4 0.004
5 0.005
6 0.006
7 0.007000000000000001
8 0.008
9 0.009
10 0.01
11 9.99985256589339e-05
12 9.999824541392405e-05
13 9.999794080037674e-05
14 9.99976118184405e-05
15 9.999725846827562e-05
16 9.999688075005433e-05
17 9.999647866396073e-05
18 9.999605221019081e-05
19 9.999560138895238e-05
```

import numpy as np


class ScheduledAdam():  # Adam optimizer + learning rate scheduler
    def __init__(self, optimizer, hidden_dim, warm_steps):
        self.init_lr = np.power(hidden_dim, -0.5)  #d_model_(-0.5)
        self.optimizer = optimizer  #optimizer= torch.optim.Adam(beta_1=0.9,beta_2=0.98, epsilon=1e-9, lr)
        self.current_steps = 0
        self.warm_steps = warm_steps # warm_steps=4000

    def step(self):
        # Update learning rate using current step information
        self.current_steps += 1
        lr = self.init_lr * self.get_scale()
        """
        warmup steps 전까지는 학습이 잘되지 않은 상태이므로 lr을 빠르게 증가시켜 변화를 크게 하고
        후에는 어느 정도의 학습이 이루어진 상태이므로 lr를 천천히 감소시켜 변화를 작게 함
        """
        for p in self.optimizer.param_groups:
            p['lr'] = lr

        self.optimizer.step()   #매개변수 갱신
     
    #갱신할 변수들에 대한 모든 변화도를 0으로 만든다.
    def zero_grad(self):
        self.optimizer.zero_grad()
    

    def get_scale(self):
        return np.min([
            np.power(self.current_steps, -0.5),
            self.current_steps * np.power(self.warm_steps, -1.5) #논문대로 -1.5로 수정
        ])

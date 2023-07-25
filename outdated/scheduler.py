# Using the Adam optimizer with a learning rate schedule as defined in the attention is all you need paper
# i.e. Learning rate increases linearly for the warmup steps and then decreases proportionally to the inverse
# square of the step number
# Commonly known as Noam Optimization

import torch.optim


def NoamScheduler(step_num, model_size=12, warmup_steps=4000):
    """
    Using the Adam optimizer with a learning rate schedule as defined in the attention is all you need paper
    Basic idea is that the learning rate increases linearly for the warmup steps and then decreases proportionally to
    the inverse square of the step number

    Args:
        model_size: Number of features in model
        step_num: Number of steps where learning rate is increasing linearly
        warmup_steps: Current step number
    """
    step_num+=1
    multiplier = min(step_num**(-0.5), step_num * warmup_steps**(-1.5))
    return model_size**(-0.5)*multiplier


if __name__=='__main__':
    """
    Example of how this would work:
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0, betas=(0.9,0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda = lambda step: noam_optimization(model_size, warmup_steps, step+1)
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
    """

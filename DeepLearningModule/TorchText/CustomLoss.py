import torch

class CustomLoss():
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def forward(self,lm_logits,targets,criterion):
        loss_poem = calc_loss_lm_logits(lm_logits, self.loss_function)
        loss = criterion(lm_logits.view(-1, vocab_size), targets.view(-1))

        return loss + loss_poem
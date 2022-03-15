import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn




import numpy as np
import matplotlib.pyplot as plt

def saliency(img, model):
    for param in model.parameters():
        param.requires_grad = False
    
    #set model in eval mode
    model.eval()
    
    input = transform(img)
    #input = input.to(device)
    #input = input.reshape(1,1,224,224)
    input.unsqueeze_(0)

     
    input.requires_grad = True
    preds = model(input)
    score, indices = torch.max(preds, 1)
    
    score.backward()
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    slc = (slc - slc.min())/(slc.max()-slc.min())

    with torch.no_grad():
        input_img = inv_normalize(input[0])
    plt.figure(figsize=(20, 10))
    plt.imshow(np.transpose(input_img.detach().numpy(), (1, 2, 0)),cmap = 'gray');
    plt.imshow(slc.numpy(), cmap=plt.cm.plasma);
    plt.show();

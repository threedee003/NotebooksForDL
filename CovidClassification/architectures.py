"""
ALexNet


"""



class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.PReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x,dim=1)

      
   #--------------------------------------------------------------------------------------------------------------------------------
  
  
  
  
"""

Custom net 1


"""




class CovNet(nn.Module):
    def __init__(self,num_classes):
        super(CovNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,kernel_size=5),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,kernel_size=4),
            #nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,kernel_size=4),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,kernel_size=4),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        
        #self.avg = nn.AdaptiveAvgPool2d((5,5))
        self.cls = nn.Sequential(
            nn.Linear(512*4*4,1000),
            #nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(1000,1000),
            #nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000,num_classes)
        )
        
    def forward(self,x):
        x = self.features(x)
        #x = self.avg(x)
        x = torch.flatten(x,1)
        x = self.cls(x)
        
        return F.log_softmax(x,dim=1)
      
      
      
      
      
      
      
      
      
      #-------------------------------------------------------------------------------------------------------------------------------------------
      
      
      
      
"""
Resnet



"""
      
class block(nn.Module):
  def __init__(
      self, in_channels, intermediate_channels, identity_downsample=None, stride=1
  ):
      super(block, self).__init__()
      self.expansion = 4
      self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
      self.bn1 = nn.BatchNorm2d(intermediate_channels)
      self.conv2 = nn.Conv2d(intermediate_channels,intermediate_channels,kernel_size=3,stride=stride,padding=1,bias=False)
      self.bn2 = nn.BatchNorm2d(intermediate_channels)
      self.conv3 = nn.Conv2d(intermediate_channels,intermediate_channels * self.expansion,kernel_size=1,stride=1,padding=0,bias=False)
      self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
      self.relu = nn.ReLU()
      self.identity_downsample = identity_downsample
      self.stride = stride

  def forward(self, x):
      identity = x.clone()

      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.conv2(x)
      x = self.bn2(x)
      x = self.relu(x)
      x = self.conv3(x)
      x = self.bn3(x)

      if self.identity_downsample is not None:
          identity = self.identity_downsample(identity)

      x += identity
      x = self.relu(x)
      return x
    
    
    
    
    
    
    
    
class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * 4,num_classes )
        self.dp = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)


        return F.log_softmax(x,dim=1)

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,intermediate_channels * 4,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
      
      
      
      
#-------------------------------------------------------------------------------------------------------------------------
      
      

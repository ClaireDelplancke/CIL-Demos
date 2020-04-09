import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient, \
                                        MaskOperator, ChannelwiseOperator
from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
                      MixedL21Norm, BlockFunction, L2NormSquared,\
                          KullbackLeibler
from ccpi.framework import TestData
import os
import sys

import numpy as np

# Specify which which type of noise to use.    
which_noise = 0
print ("which_noise ", which_noise)

# Specify whether to do gray (0), colour using channelwise (1), colour mask (2)
colour_mode = 1

# Load in test image
loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data_rgb = loader.load(TestData.PEPPERS)
ig_rgb = data_rgb.geometry

# Create gray version of image
data_gray = 0.2989*data_rgb.subset(channel=0) + \
            0.5870*data_rgb.subset(channel=1) + \
            0.1140*data_rgb.subset(channel=2)
ig_gray = data_gray.geometry


# Create mask with letters CIL
if colour_mode < 2:
    mask = ig_gray.allocate(True,dtype=np.bool)
else:
    mask = ig_rgb.allocate(True,dtype=np.bool)
amask = mask.as_array()

# Letter C
amask[50:-50,56-10:56+10] = False
amask[-70:-50,56-10:166] = False
amask[50:70,56-10:166] = False

# Letter I
amask[50:-50,256-10:256+10] = False
amask[50:70,256-50:256+50] = False
amask[-70:-50,256-50:256+50] = False

# Letter L
amask[50:-50,356-10:356+10] = False
amask[-70:-50,356-10:466] = False

# If 0, use the gray version and do a single-channel mask and inpainting.
# If 1, use ChannelwiseOperator to do colour inpainting.
# If 2, use multi-channel mask in MaskOperator to do colour inpainting.
if colour_mode==0:
    MO = MaskOperator(mask)
    data = data_gray
elif colour_mode==1:
    MO = ChannelwiseOperator(MaskOperator(mask),3,'append')
    data = data_rgb
else:
    MO = MaskOperator(mask)
    data = data_rgb
ig = data.geometry

# Create noisy and masked data: First add noise, then mask the image with 
# MaskOperator.
noises = ['gaussian', 'poisson', 's&p']
noise = noises[which_noise]
if noise == 's&p':
    n1 = TestData.random_noise(data.as_array(), mode = noise, salt_vs_pepper = 0.9, amount=0.2)
elif noise == 'poisson':
    scale = 5
    n1 = TestData.random_noise( data.as_array()/scale, mode = noise, seed = 10)*scale
elif noise == 'gaussian':
    n1 = TestData.random_noise(data.as_array(), mode = noise, seed = 10)
else:
    raise ValueError('Unsupported Noise ', noise)
noisy_data = ig.allocate()
noisy_data.fill(n1)

noisy_data = MO.direct(noisy_data)    

# Regularisation Parameter depending on the noise distribution
if noise == 's&p':
    alpha = 0.8
elif noise == 'poisson':
    alpha = 1.0
elif noise == 'gaussian':
    alpha = .1

# Choose data fidelity dependent on noise type.
if noise == 's&p':
    f2 = L1Norm(b=noisy_data)
elif noise == 'poisson':
    f2 = KullbackLeibler(noisy_data)
elif noise == 'gaussian':
    f2 = 0.5 * L2NormSquared(b=noisy_data)

# Create operators
op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)
op2 = MO

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Create functions      
f = BlockFunction(alpha * MixedL21Norm(), f2) 
g = ZeroFunction()
        
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 100
pdhg.run(1000)

# Show results
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.imshow(data.as_array(),vmin=0.0,vmax=1.0)
plt.title('Ground Truth')
if colour_mode==0:
    plt.gray()
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(noisy_data.as_array(),vmin=0.0,vmax=1.0)
plt.title('Noisy and Masked Data')
if colour_mode==0:
    plt.gray()
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(pdhg.get_output().as_array(),vmin=0.0,vmax=1.0)
plt.title('TV Reconstruction')
if colour_mode==0:
    plt.gray()
plt.colorbar()
plt.show()

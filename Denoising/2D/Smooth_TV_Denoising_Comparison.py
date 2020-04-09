import numpy as np 
import numpy                          
import matplotlib.pyplot as plt


from ccpi.optimisation.operators import Gradient
from ccpi.optimisation.functions import SmoothMixedL21Norm,  L2NormSquared, FunctionOperatorComposition, MixedL21Norm
from ccpi.optimisation.algorithms import GradientDescent, FISTA, PDHG

from ccpi.framework import TestData
import os
import sys

# Load ground truth and define geometries
loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(512,512))
ig = data.geometry
ag = ig

# Add Gaussian Noise 
n1 = TestData.random_noise(data.as_array(), mode = 'gaussian', seed = 10)
noisy_data = ig.allocate()
noisy_data.fill(n1)

# Setup Gradient Descent algorithm, with smooth approximation of TV
alpha = 3
epsilon = 1e-6
Grad = Gradient(ig)
f1 = FunctionOperatorComposition( alpha * SmoothMixedL21Norm(epsilon), Grad)
f2 = 0.5 * L2NormSquared(b=noisy_data)
objective_function = f1  +  f2
step_size = 0.00001

print("Running Gradient Descent with smooth approximation of TV\n This will take some time .... ")
x_init = noisy_data
gd = GradientDescent(x_init, objective_function, step_size,
                     max_iteration = 10000,update_objective_interval = 500)
gd.run(verbose=True)


# Show results
plt.figure(figsize=(20,5))
plt.subplot(1,4,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,4,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(1,4,3)
plt.imshow(gd.get_output().as_array())
plt.title('GD Reconstruction')
plt.colorbar()
plt.subplot(1,4,4)
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'GTruth')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), gd.get_output().as_array()[int(ig.shape[0]/2),:], label = 'TV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()


print("Running FISTA with smooth approximation of TV\n This will take some time .... ")
# Setup and run FISTA algorithm
objective_function.L = 1000000
fi = FISTA(x_init, objective_function, max_iteration = 10000,update_objective_interval = 1000)
fi.run(verbose=True)


## Show results
plt.figure(figsize=(20,5))
plt.subplot(1,4,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,4,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(1,4,3)
plt.imshow(fi.get_output().as_array())
plt.title('FISTA Reconstruction')
plt.colorbar()
plt.subplot(1,4,4)
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'GTruth')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), fi.get_output().as_array()[int(ig.shape[0]/2),:], label = 'TV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()


print("Running PDHG with non-smooth TV")

# Setup and run PDHG for the non-smooth TV

method = 1
operator = Grad
f =  alpha * MixedL21Norm()
g = f2
        
    
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)

#%%
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma,
            max_iteration = 10000,
            update_objective_interval = 1000)
pdhg.run()

## Show results
plt.figure(figsize=(20,5))
plt.subplot(1,4,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,4,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(1,4,3)
plt.imshow(pdhg.get_output().as_array())
plt.title('PDHG Reconstruction')
plt.colorbar()
plt.subplot(1,4,4)
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'GTruth')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(ig.shape[0]/2),:], label = 'TV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()


# Plot objective values
plt.figure()
plt.loglog(gd.objective,label = 'GD')
plt.loglog(fi.objective,label = 'FISTA')
pdhg_obj = []
pdhg_dobj = []
for sublist in pdhg.objective:
    pdhg_obj.append(sublist[0])
    pdhg_dobj.append(sublist[1])
plt.loglog(pdhg_obj,label = 'PDHG primal')
plt.loglog(pdhg_dobj,label = 'GTruth dual')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('objective')
plt.show()
#========================================================================
# Copyright 2017-2020 Science Technology Facilities Council
# Copyright 2017-2020 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================


import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.operators import Gradient
from ccpi.optimisation.functions import SmoothMixedL21Norm,  L2NormSquared, \
                                     FunctionOperatorComposition, MixedL21Norm
from ccpi.optimisation.algorithms import GradientDescent, FISTA, PDHG

from ccpi.framework import TestData
import os
import sys

#%% Load test image and add noise
loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(512,512))
ig = data.geometry
ag = ig

n1 = TestData.random_noise(data.as_array(), mode = 'gaussian', seed = 10)

noisy_data = ig.allocate()
noisy_data.fill(n1)

#%% Set up least squares plus smoothed TV regularisation denoising problem 
#   and solve with Gradient Descent with constant step size.

# TV regularisation and smoothing parameters
alpha = 3
epsilon = 1e-6

# Smooth TV functional
Grad = Gradient(ig)
f1 = FunctionOperatorComposition( alpha * SmoothMixedL21Norm(epsilon), Grad)

# Least squares denoising functional (no forward operator)
f2 = 0.5 * L2NormSquared(b=noisy_data)

# Sum two smooth functionals together
objective_function = f1  +  f2

# Set algorithm parameters
step_size = 0.00002
x_init = noisy_data

# Set up and run Gradient Descent algorithm
print("Running Gradient Descent with smooth approximation of TV.\nThis will take some time .... ")
gd = GradientDescent(x_init, objective_function, step_size=step_size,
                     max_iteration = 10000,update_objective_interval = 100)
gd.run(verbose=True)

## Show Gradient Descent reconstruction results
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


#%%  Use FISTA algorithm to solve same smoothed TV regularised denoising proglem

# Manually set guess of Lipschitz parameter of function to step size selection.
objective_function.L = 1000000

# Set up and run FISTA algorithms
print("Running FISTA with smooth approximation of TV.\nThis will take some time .... ")
fi = FISTA(x_init, objective_function, max_iteration = 10000, 
           update_objective_interval = 100)
fi.run(verbose=True)

## Show FISTA reconstruction results
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


#%% Use PDHG to solve non-smooth version of problem for comparison

# Set up non-smooth TV regularisation term
operator = Grad
f =  alpha * MixedL21Norm()
        
# Set algorithm parameters: primal and dual step sizes, sigma and tau, 
# standard choices based on operator's norm.
normK = operator.norm()
sigma = 1
tau = 1/(sigma*normK**2)

# Setup and run the PDHG algorithm
print("Running PDHG with non-smooth TV.\nThis will take some time...")
pdhg = PDHG(f=f,g=f2,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 10000
pdhg.update_objective_interval = 100
pdhg.run(very_verbose=True)

## Show PDHG reconstruction results
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

#%%

# Plot convergence of GD, FISTA and for PDHG both primal, dual objectives 
# and the duality gap. The primal and dual  objective should converge to 
# each other and the gap, which is the difference between the two, should
# approach zero, which it can be seen to do.
plt.figure()
plt.loglog(gd.objective,label = 'GD')
plt.loglog(fi.objective,label = 'FISTA')
plt.loglog(pdhg.objective,label = 'PDHG primal')
plt.loglog(pdhg.dual_objective,label = 'PDHG dual')
plt.loglog(pdhg.primal_dual_gap,label = 'PDHG gap')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('objective')
plt.show()


# This demo illustrates how ASTRA 2D projectors can be used with
# the modular optimisation framework. The demo sets up a 2D test case and 
# demonstrates reconstruction using CGLS, as well as FISTA for least squares 
# and 1-norm regularisation.

# First make all imports
from ccpi.framework import ImageData , ImageGeometry, AcquisitionGeometry
from ccpi.optimisation.algorithms import FISTA, CGLS
from ccpi.optimisation.functions import LeastSquares, L1Norm, ZeroFunction
from ccpi.astra.operators import AstraProjectorSimple
from ccpi.astra.processors import FBP

import numpy as np
import matplotlib.pyplot as plt

import os
import tomophantom
from tomophantom import TomoP2D


model = 1 # select a model number from the library
N = 128 # set dimension of the phantom
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")

phantom_2D = TomoP2D.Model(model, N, path_library2D)    
data = ImageData(phantom_2D)
ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N)

# Create acquisition data and geometry
detectors = int(np.sqrt(2)*N) 
angles = np.linspace(0, np.pi, 90)
ag = AcquisitionGeometry('parallel','2D',angles, detectors)


# Select device
device = input('Available device: GPU==1 / CPU==0 ')
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'
        
# Select geometry

angles_num = 90
det_w = 1.0
det_num = N
SourceOrig = 200
OrigDetec = 0    
    
test_case = input('Geometry: Parallel==1 / Cone==0 ')

if test_case=='1':
    angles = np.linspace(0,np.pi,angles_num,endpoint=False)
    ag = AcquisitionGeometry('parallel',
                             '2D',
                             angles,
                             det_num)
elif test_case=='0':
    angles = np.linspace(0,np.pi,angles_num,endpoint=False)
    ag = AcquisitionGeometry('cone',
                             '2D',
                             angles,
                             det_num,
                             dist_source_center=SourceOrig, 
                             dist_center_detector=OrigDetec)      
else:
    NotImplemented    


Aop = AstraProjectorSimple(ig, ag, dev)
sin = Aop.direct(data)

# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,10))
plt.subplot(1,2,2)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,2,1)
plt.imshow(sin.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.show()


# Filtered back projection
fbp = FBP(ig, ag, filter_type = 'ram-lak', device = dev)
fbp.set_input(sin)
fbp_recon = fbp.get_output()
plt.imshow(fbp_recon.as_array())
plt.colorbar()
plt.title('Filtered BackProjection')
plt.show()
#%%


x_init = ig.allocate()
# First a CGLS reconstruction can be done:
CGLS_alg = CGLS(x_init=x_init, operator=Aop, data=sin,
                max_iteration = 1000,
                update_objective_interval = 200)
CGLS_alg.run(verbose = True)

x_CGLS = CGLS_alg.get_output()

plt.figure()
plt.imshow(x_CGLS.array)
plt.title('CGLS')
plt.show()

plt.figure()
plt.semilogy(CGLS_alg.objective)
plt.title('CGLS criterion')
plt.show()

# CGLS solves the simple least-squares problem. The same problem can be solved 
# by FISTA by setting up explicitly a least squares function object and using 
# no regularisation:

# Create least squares object instance with projector, test data and a constant 
# coefficient of 0.5:
f = LeastSquares(A=Aop,b=sin,c=1)
#f= FunctionOperatorComposition(L2NormSquared(b=b),Aop)
# Run FISTA for least squares without constraints

FISTA_alg = FISTA(x_init=x_init, f=f, g=ZeroFunction(),
                  max_iteration = 2000)
FISTA_alg.run(verbose=False)
x_FISTA = FISTA_alg.get_output()

plt.figure()
plt.imshow(x_FISTA.as_array())
plt.title('FISTA Least squares reconstruction')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(FISTA_alg.objective)
plt.title('FISTA Least squares criterion')
plt.show()


# FISTA can also solve regularised forms by specifying a second function object
# such as 1-norm regularisation with choice of regularisation parameter lam:

# Create 1-norm function object
lam = 1.0
g0 = lam * L1Norm()

# Run FISTA for least squares plus 1-norm function.
FISTA_alg1 = FISTA(x_init=x_init, f=f, g=g0,max_iteration = 2000)
FISTA_alg1.run(verbose=False)
x_FISTA1 = FISTA_alg1.get_output()

plt.figure()
plt.imshow(x_FISTA1.array)
plt.title('FISTA LS+L1Norm reconstruction')
plt.colorbar()
plt.show()

plt.figure()
plt.semilogy(FISTA_alg1.objective)
plt.title('FISTA LS+L1norm criterion')
plt.show()

#%%

# Compare all reconstruction and criteria
clims = (0,1)
cols = 2
rows = 2
current = 1

fig = plt.figure()
a=fig.add_subplot(rows,cols,current)
a.set_title('phantom {0}'.format(np.shape(data.as_array())))
imgplot = plt.imshow(data.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('CGLS')
imgplot = plt.imshow(x_CGLS.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA LS')
imgplot = plt.imshow(x_FISTA.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA LS+1')
imgplot = plt.imshow(x_FISTA1.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

fig = plt.figure()
a=fig.add_subplot(1,1,1)
a.set_title('criteria')
imgplot = plt.loglog(CGLS_alg.objective, label='CGLS')
imgplot = plt.loglog(FISTA_alg.objective , label='FISTA LS')
imgplot = plt.loglog(FISTA_alg1.objective , label='FISTA LS+1')
a.legend(loc='lower left')
plt.show()

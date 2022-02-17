from ANNarchy import *
import pylab as plt
from BGM_22 import izhikevich2003
setup(dt=0.1)


"""
README:

1. set a,b,c,d
if b < 0.267:
1. calculate v1,v2,u1,u2 = steady-state points
2. initialize u,v with steady-state points
3. check if its a stable steady-state (small stimulus --> goes back to steady-state?)
4. if yes ... great
5. if no... init v and u within phase plot
6. init v and u at minimum v
else:
1. init v and u with 0
2. check phase plot
3. init v and u within phase plot
4. init v and u at minimum v

"""

a=0.005
b=0.265
c=-65
d=2

if b<0.267:
    v1=-(5**(3/2)*np.sqrt(5*b**2-50*b+13)-25*b+125)/2
    v2=(5**(3/2)*np.sqrt(5*b**2-50*b+13)+25*b-125)/2
    u1=b*v1
    u2=b*v2
    print('b:',b)
    print('v1:',v1)
    print('u1:',u1)
    print('v2:',v2)
    print('u2:',u2)


pop=Population(1,neuron=izhikevich2003)
pop.a=a
pop.b=b
pop.c=c
pop.d=d

m=Monitor(pop,['v','u'])

compile()
### INITIALIZE HERE
pop.v=-69.41572635560657
pop.u=-14.338099488135063

simulate(1000)
pop.I_add=10
simulate(1)
pop.I_add=0
simulate(3000)

v=m.get('v')[:,0]
u=m.get('u')[:,0]

plt.figure()
plt.plot(v,u)
plt.plot(v[0],u[0],'k.',markersize=20)
plt.savefig('phase_plot.svg')

plt.figure()
plt.subplot(211)
plt.plot(v,'k')
plt.subplot(212)
plt.plot(u,'k')
plt.savefig('time_plot.svg')

print(v[np.argmin(v)])
print(u[np.argmin(v)])





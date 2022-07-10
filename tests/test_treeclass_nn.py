
from kernex import treeclass
import jax 
import jax.numpy as jnp 
import numpy as np 
from typing import Sequence,Callable

def test_nn():
  @treeclass
  class Linear :

    weight : jnp.ndarray
    bias   : jnp.ndarray

    def __init__(self,key,in_dim,out_dim):
      self.weight = jax.random.normal(key,shape=(in_dim, out_dim)) * jnp.sqrt(2/in_dim)
      self.bias = jnp.ones((1,out_dim))
    
    def __call__(self,x):
      return x @ self.weight + self.bias

  @treeclass
  class StackedLinear:
    layers : Sequence[Linear]

    def __init__(self,key,layers):
      keys= jax.random.split(key,len(layers)-1)

      self.layers = []

      for ki,in_dim,out_dim in zip(keys,layers[:-1],layers[1:]) :
        self.layers += [Linear(ki,in_dim,out_dim)]
    
    def __call__(self,x):
      for layer in self.layers[:-1]:
        x = layer(x)
        x = jax.nn.tanh(x)

      return self.layers[-1](x)
    
    def apply(self,x):
      return x
    

  x = jnp.linspace(0,1,100)[:,None]
  y = x**3 + jax.random.uniform(jax.random.PRNGKey(0),(100,1))*0.01

  model = StackedLinear(layers=[1,128,128,1],key=jax.random.PRNGKey(0))

  def loss_func(model,x,y):
    return jnp.mean((model(x)-y)**2 ) 

  @jax.jit
  def update(model,x,y):
    value,grads = jax.value_and_grad(loss_func)(model,x,y)
    return value , jax.tree_map(lambda x,y:x-1e-3*y,model,grads)

  for _ in range(1,2001):
    value,model = update(model,x,y)
  
  np.testing.assert_allclose(value, jnp.array(0.00103019),atol=1e-5)



def test_nn_with_func_input():

  @treeclass
  class Linear :

    weight : jnp.ndarray
    bias   : jnp.ndarray
    act_func : Callable

    def __init__(self,key,in_dim,out_dim,act_func):
      self.act_func = act_func
      self.weight = jax.random.normal(key,shape=(in_dim, out_dim)) * jnp.sqrt(2/in_dim)
      self.bias = jnp.ones((1,out_dim))
    
    def __call__(self,x):
      return self.act_func(x @ self.weight + self.bias)

  layer = Linear(key=jax.random.PRNGKey(0), in_dim=1, out_dim=1, act_func= jax.nn.tanh)
  x = jnp.linspace(0,1,100)[:,None]
  y = x**3 + jax.random.uniform(jax.random.PRNGKey(0),(100,1))*0.01
  layer(x)
  return True

import theano
from theano import *
import theano.tensor as T

##baby step 1
####
### {b, w, i, l, f, d, c}{scalar, vector, matrix, row, col, tensor3, tensor4}
### tensor.
#####
x = T.dscalar('x')
y = T.dscalar('y')
z = x ** 2 + y ** 2

fun = function(inputs=[x, y], z)

fun(1, 2) ##returns array(5)

###besides
x, y, z = T.dmatrices(3)
xx, yy, zz = T.dmatrices('x', 'y', 'z')  #### complains if the number of names and tensors don't match


##build in function of tensor

logistic = 1 / (1.0 + T.exp(-x))

###default value
##make use of In class
x, y, w = T.dscalars('x', 'y', 'z')
z = (x + y) * w
f = function([x, In(y, 1), In(w, value=2, name='w_by_name')], z)
###In class  change the parameter name to name='w_by_name'
## so call the function by f(x=1, y=2, w_by_name=3)


###shared variables
##hyper symbolic and non-symbolic variables, value may be shared by multi-functions
##from
## x = shared(value, name=None, strict=False, allow_downcast=None, **kwargs)
x = shared(0)
x.get_value()
x.set_value(1)

state = shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates=[(state, state+inc)])
###how to copy functions? how to use shared variables with updating its values and how to use it with another value??

##logistic functions


###random numbers



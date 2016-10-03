import numpy as np
import matplotlib.pyplot as plt

def sarcos_demo():
    training=np.genfromtxt("sarcos_inv.csv",delimiter=",")
    test=np.genfromtxt("sarcos_inv_test.csv",delimiter-",")
    return ()

def marathon_demo(length_scale=30,noise=0.5):

    import pods
    data=pods.datasets.olympic_marathon_men()
    y=data["Y"] #size n
    x_train=data["X"] #size n
    y_train=y-y.mean()
    #makeup test set
    x_test=np.linspace(x_train.min(),x_train.max(),100) #size n*
    x_test.resize((100,1))
    fstar,cov,plogy=GPR_simple(x_train,y_train,x_test,length_scale,noise)
    y_train=y_train+y.mean()
    fstar=fstar+y.mean()
    fig=plot_simple_demo(x_train,y_train,x_test,fstar,cov)
    return (x_test,fstar,cov,fig)

def sin_demo(length_scale=1,noise=0):
    sigma2e=0.001
    x_train=np.random.normal(0,2,(10,1))
    y_train=np.sin(2*np.pi*x_train)
    x_test=np.linspace(-5,5,100)
    x_test.resize((100,1))
    fstar,cov,plogy=GPR_simple(x_train,y_train,x_test,length_scale,noise=sigma2e)
    fig=plot_simple_demo(x_train,y_train,x_test,fstar,cov)
    return (x_test,fstar,cov,fig)


def plot_simple_demo(x_train,y_train,x_test,fstar,cov):
    #scale around confidence interval
    var=np.diag(cov)
    se=2*np.sqrt(var.reshape((np.max(var.shape),1)))
    yvect=np.vstack((y_train,fstar+se,fstar-se))
    y_range=np.vstack((yvect.max(),yvect.min()))
    xvect=np.vstack((x_train,x_test))
    x_range=np.vstack((xvect.max(),xvect.min()))

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x_train,y_train,marker="+",color="b")
    ax.plot(x_test,fstar+se,color="r")
    ax.plot(x_test,fstar-se,color="r")
    ax.plot(x_test,fstar,color="b")
    top=fstar+se
    bot=fstar-se
    plt.fill_between(x_test[:,0],top[:,0],bot[:,0],color='blue',alpha=0.3)
    fig.show()
    return (fig)


def GPR_simple(x_train,y_train,x_test,length_scale=None,noise=None):
    kern=RBF
    #perform convolution function, i.e. find k,k*,K*
    K=kern(x_train,x_train,length_scale,sigma=1) #size n by n
    if noise is None:
        noise=0

    #do a matrix inversion using cholesky
    L=np.linalg.cholesky(K+np.square(noise)*np.identity(x_train.shape[0])) #size n by n
    alpha=np.linalg.solve(np.transpose(L),np.linalg.solve(L,y_train)) #size n by 1
    print("Inversion of L successful")

    #perform covariance function against x_train and x_test
    k_star=kern(x_train,x_test,length_scale,sigma=1) #size n by n*

    #each mean point is a linear combination of the kernels wrt the weight
    fstar=np.dot(np.transpose(k_star),alpha) #size n*
    print("Predicted mean computed")

    #calculate the new variance
    v=np.linalg.lstsq(L,k_star)[0] #size n* by n*
    #calculate K_star, covariance function between the two inputs
    K_star=kern(x_data=x_test,x_data_star=x_test,length_scale=length_scale,sigma=1)

    #calculate the predictive variance between the points
    var=K_star-np.dot(np.transpose(v),v)
    print("Predicted variance computed")

    #calculated the logrithmic marginal likelihood
    plogy=-0.5 * np.dot(np.transpose(y_train),alpha) - np.sum(np.log(np.diag(L)))-x_train.shape[0]/2*np.log(2*np.pi)

    return(fstar,var,plogy)

def RBF(x_data,x_data_star,length_scale=None,sigma=1):
    """
    Radial Basis kernel (gaussian)
    to be optimized... lolz
    can take in multidimensional data set
    each row is a data point, each column is a dimension
    """
    I=x_data.shape[0]
    if length_scale is None:
        length_scale=1
    print(length_scale)
    J=x_data_star.shape[0]
    K=np.zeros((I,J))
    """
    the below can be faster by using pop, but we will do that later
    """
    for i in range(I):
        for j in range(J):
            r=np.linalg.norm(x_data[i,:]-x_data_star[j,:])
            K[i,j]=np.square(sigma)*np.exp(-(np.square(r))/(2*np.square(length_scale)))
    return(K)

def periodic(x_data,x_data_star,length_scale=None,sigma=1):
    """
    sin kernel
    to be optimized... lolz
    can take in multidimensional data set
    each row is a data point, each column is a dimension
    """
    I=x_data.shape[0]
    if length_scale is None:
        length_scale=1

    J=x_data_star.shape[0]
    K=np.zeros((I,J))
    p=1
    """
    the below can be faster by using pop, but we will do that later
    """
    for i in range(I):
        for j in range(J):
            r=np.linalg.norm(x_data[i,:]-x_data_star[j,:])
            K[i,j]=np.square(sigma)*np.exp(-2*np.square(np.pi*np.sin(r)/p)/np.square(length_scale))
    return(K)

def sink(x_data,x_data_star,length_scale=None,sigma=1):
    """
    sin kernel
    to be optimized... lolz
    can take in multidimensional data set
    each row is a data point, each column is a dimension
    """
    I=x_data.shape[0]
    if length_scale is None:
        length_scale=1

    J=x_data_star.shape[0]
    K=np.zeros((I,J))
    """
    the below can be faster by using pop, but we will do that later
    """
    for i in range(I):
        for j in range(J):
            if i==j:
                K[i,j]=1
            else:
                r=np.linalg.norm(x_data[i,:]-x_data_star[j,:])
                K[i,j]=np.sin(r)/r
    return(K)



def linear(x_data,x_data_star,sigmab=1,sigmav=1):
    """
    sin kernel
    to be optimized... lolz
    can take in multidimensional data set
    each row is a data point, each column is a dimension
    """
    I=x_data.shape[0]
    if length_scale is None:
        length_scale=1

    J=x_data_star.shape[0]
    K=np.zeros((I,J))
    """
    the below can be faster by using pop, but we will do that later
    """
    c=0
    for i in range(I):
        for j in range(J):
            r=(x_data[i,:]-c)*(x_data_star[j,:]-c)
            K[i,j]=np.square(sigmab)+np.square(sigmav)*r
    return(K)



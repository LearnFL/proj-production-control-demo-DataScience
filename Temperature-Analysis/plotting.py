import matplotlib.pyplot as plt

def plot10(x:float, a10):
    a = plt.plot([a10["DateTime"].values[1],a10["DateTime"].values[-1]], [x[0] - 0.5, x[1] - 0.5] , color="r", lw=1) 
    b = plt.plot([a10["DateTime"].values[1],a10["DateTime"].values[-1]], [x[0] + 0.5, x[1] + 0.5], color="r", lw=1) 
    return a, b

def plot20(x:float, a20):
    a = plt.plot([a20["DateTime"].values[1],a20["DateTime"].values[-1]], [x[1] - 0.5, x[2] - 0.5] , color="r", lw=1) 
    b = plt.plot([a20["DateTime"].values[1],a20["DateTime"].values[-1]], [x[1] + 0.5, x[2] + 0.5], color="r", lw=1) 
    return a, b

def plot30(x:float, a30):
    a = plt.plot([a30["DateTime"].values[1],a30["DateTime"].values[-1]], [x[2] - 0.5, x[3] - 0.5] , color="r", lw=1) 
    b = plt.plot([a30["DateTime"].values[1],a30["DateTime"].values[-1]], [x[2] + 0.5, x[3] + 0.5], color="r", lw=1) 
    return a, b

def plot40(x:float, a40):
    a = plt.plot([a40["DateTime"].values[1],a40["DateTime"].values[-1]], [x[3] - 0.5, x[4] - 0.5] , color="r", lw=1) 
    b = plt.plot([a40["DateTime"].values[1],a40["DateTime"].values[-1]], [x[3] + 0.5, x[4] + 0.5], color="r", lw=1) 
    return a, b

def plot50(x:float, a50):   
    a = plt.plot([a50["DateTime"].values[1],a50["DateTime"].values[-1]], [x[4] - 0.5, x[5] - 0.5] , color="r", lw=1) 
    b = plt.plot([a50["DateTime"].values[1],a50["DateTime"].values[-1]], [x[4] + 0.5, x[5] + 0.5], color="r", lw=1) 
    return a, b

def plot60(x:float, a60):
    a = plt.plot([a60["DateTime"].values[1],a60["DateTime"].values[-1]], [x[5] - 0.5, x[6] - 0.5] , color="r", lw=1) 
    b = plt.plot([a60["DateTime"].values[1],a60["DateTime"].values[-1]], [x[5] + 0.5, x[6] + 0.5], color="r", lw=1) 
    return a, b
        
def plot70(x:float, a70):
    a = plt.plot([a70["DateTime"].values[1],a70["DateTime"].values[-1]], [x[6] - 0.5, x[7] - 0.5] , color="r", lw=1) 
    b = plt.plot([a70["DateTime"].values[1],a70["DateTime"].values[-1]], [x[6] + 0.5, x[7] + 0.5], color="r", lw=1) 
    return a, b

def plot80(x:float, a80):
    a = plt.plot([a80["DateTime"].values[1],a80["DateTime"].values[-1]], [x[7] - 0.5, x[8] - 0.5] , color="r", lw=1) 
    b = plt.plot([a80["DateTime"].values[1],a80["DateTime"].values[-1]], [x[7] + 0.5, x[8] + 0.5], color="r", lw=1) 
    return a, b

func_list = [plot10, plot20, plot30, plot40, plot50, plot60, plot70, plot80]

class Plotter:
    def __init__(self, period,  func_list=func_list):
        self.func_list = func_list
        self.period = period
    
    def __call__(self, limit):
        for func, per in zip(self.func_list, self.period):
            try:
                func(limit, per)
            except:
                pass 

    def __repr__(self) -> str:
        return f'{self.__class__.__name__!r}, Used to plot temperature range for graphs, {func_list!r}'
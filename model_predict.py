import numpy as np
from scipy.optimize import minimize_scalar

class model_predictor:
    def __init__(self, gpu_mem):
        self.gpu_mem = gpu_mem
    def memory_usage(self, layer, batch):
        m_s = 16 * 12 * 128**2 * layer**3
        m_d = batch * 5 * 128**2 * 8 * layer**2
        return m_s + m_d

    def objective(self, layer, batch):
        if layer < 0:
            return np.inf
        mem_used = self.memory_usage(layer, batch)
        if mem_used > self.gpu_mem:
            return np.inf 
        return -layer 

    def optimize_l(self, batch):
        res = minimize_scalar(self.objective, args=batch, bounds=(0, 100), method='bounded')
        
        if res.success and res.fun != np.inf:
            return res.x
        else:
            return "No feasible solution found that satisfies the GPU memory constraint."
        
    def optimize_coin(self, batch):
        from pulp import LpProblem, LpMinimize, LpVariable, lpSum, COIN_CMD
        # Define the problem
        prob = LpProblem("Layer_Optimization", LpMinimize)
        
        # Define the decision variable
        layer = LpVariable("layer", lowBound=0, upBound=100, cat='Continuous')
        
        # Define the objective function
        prob += -layer
        
        # Define the constraints
        mem_used = self.memory_usage(layer, batch)
        prob += mem_used <= self.gpu_mem
        
        # Solve the problem using COIN-OR CBC
        prob.solve(COIN_CMD())
        
        if prob.status == 1:
            return layer.varValue
        else:
            return "No feasible solution found that satisfies the GPU memory constraint."


def main():
    gpu_mem_value = 576 * 10**9
    predictor = model_predictor(gpu_mem_value)
    batch_sizes = [8]
    results = {}

    for batch in batch_sizes:
        result = predictor.optimize_l(batch)
        # result = predictor.optimize_coin(batch)
        results[batch] = result
        print(f"For batch size {batch}, the largest possible value of 'layer' is approximately: {result}")
    
    
def func(x):
    return x**2+2*x+1

def simple_test():    
    min_f_x = minimize_scalar(func, bounds=(0, 10), method='bounded')
    print(f'The minimum of the function is at x={min_f_x.x:.2f} with value f(x)={func(min_f_x.x):.2f}, with bound (0, 10)')
    min_f_x = minimize_scalar(func)
    print(f'The minimum of the function is at x={min_f_x.x} with value f(x)={func(min_f_x.x)}')
    
def simple_coin():
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, COIN_CMD
    # Define the problem
    prob = LpProblem("func_Optimization", LpMinimize)
    
    # Define the decision variable
    var = LpVariable("x", lowBound=-10, upBound=100, cat='Continuous')
    
    # Define the objective function
    prob += -func(var.varValue)
    
    # Define the constraints
    
    prob += var >= 0
    
    # Solve the problem using COIN-OR CBC
    prob.solve(COIN_CMD())
    
    if prob.status == 1:
        return var.varValue
    else:
        return "No feasible solution found that satisfies the GPU memory constraint."
if __name__ == "__main__":
    main()
    # simple_test()
    # simple_coin()

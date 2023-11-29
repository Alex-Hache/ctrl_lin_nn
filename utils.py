import torch

def find_module(model, target_class):
    for name, module in model.named_children():
        if isinstance(module, target_class):
            print(f"Le module '{name}' est une instance de la classe '{target_class.__name__}'.")
            return module
        if list(module.children()):
            found_module = find_module(module, target_class)
            if found_module is not None:
                return found_module
    return None
         
def block_diag(Mat):
    n = sum(M.shape[0] for M in Mat)
    m = sum(M.shape[1] for M in Mat)

    A = torch.zeros((n, m))

    index1 = 0
    index2 = 0
    for M in Mat:
        A[index1:index1 + M.shape[0], index2:index2 + M.shape[1]] = M
        index1 += M.shape[0]
        index2 += M.shape[1]
    return A

def is_legal(v):
    legal = not torch.isnan(v).any() and not torch.isinf(v)
    return legal
    
def bmat(Mat):
    Mix = [torch.cat(Mij, dim=1) for Mij in Mat]
    return torch.cat(Mix, dim=0)

def isSDP(L):
    '''
    L is tensor
    '''
    eigval, _ = torch.linalg.eig(L)
    isAllEigPos = torch.all(torch.real(eigval)>0) 
    isSymetric = torch.all(L == L.T)
    if not isAllEigPos:
        print("Not all eigenvalues are positive")
    if not isSymetric:
        print("Matrix is not symmetric")
    bSDP = isSymetric and isAllEigPos
    return bSDP

def getEigenvalues(L):
    '''
        params :
            - L pytorch Tensor
    '''
    eigval, _ = torch.linalg.eig(L)
    return eigval

def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    vals, vecs = torch.eig(matrix, eigenvectors=True)
    vals = torch.view_as_complex(vals.contiguous())
    vals_pow = vals.pow(p)
    vals_pow = torch.view_as_real(vals_pow)[:, 0]
    matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs)))
    return matrix_pow
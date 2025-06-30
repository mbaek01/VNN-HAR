import torch

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def extract_norm(x: torch.Tensor, p=2, dim=1, keepdim=True) -> torch.Tensor:
    """
    Extracts L2 norm from each row in Lx3 tensor.
    """
    return torch.norm(x, p=p, dim=dim, keepdim=keepdim)


class NormSVMClassifier:
    def __init__(self, C=1.0, kernel='rbf'):
        self.model = make_pipeline(StandardScaler(), SVC(C=C, kernel=kernel))

    def fit(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor):
        """
        x_tensor: torch.Tensor of shape (L, 3)
        y_tensor: torch.Tensor of shape (L,)
        """
        x_norm = extract_norm(x_tensor)       # (L, 1)
        X_np = x_norm.numpy()
        y_np = y_tensor.numpy()
        self.model.fit(X_np, y_np)

    def predict(self, x_tensor: torch.Tensor):
        x_norm = extract_norm(x_tensor)
        return self.model.predict(x_norm.numpy())

    def score(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor):
        x_norm = extract_norm(x_tensor)
        return self.model.score(x_norm.numpy(), y_tensor.numpy())
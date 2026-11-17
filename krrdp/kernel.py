from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


def rbf(x, z, lengthscale):
    return np.exp(cdist(x, z, "sqeuclidean") * (-0.5/lengthscale**2))


@dataclass
class KRR:
    centers: np.ndarray
    coefficients: np.ndarray
    lengthscale: float
    clip: float
    solver: str

    @classmethod
    def fit(cls, x, y, lengthscale=80., regularization=1e-6, clip=100.,
            max_centers=None, rng=None):
        """Minimize mean squared loss + lambda ||f||_H^2.

        Full dual system is (K + n*lambda*I)a=y. The optional Nyström
        restriction is a direct Cholesky solve, not the FALKON CG solver.
        """
        n = len(x)
        if np.all(x == x[0]):
            # Exact solution for coincident training points, including time zero.
            return cls(x[:1].copy(), np.atleast_1d(y.mean()/(1+regularization)),
                       lengthscale, clip, "degenerate_exact")
        if max_centers is None or max_centers >= n:
            kernel = rbf(x, x, lengthscale)
            kernel.flat[::n+1] += n*regularization
            coefficients = cho_solve(cho_factor(kernel, lower=True), y)
            return cls(x.copy(), coefficients, lengthscale, clip, "full")
        if rng is None:
            raise ValueError("An explicit RNG is required for Nyström sampling")
        centers = x[rng.choice(n, max_centers, replace=False)].copy()
        # Whiten the Nyström basis to avoid squaring its condition number.
        kmm = rbf(centers, centers, lengthscale)
        eigenvalues, eigenvectors = np.linalg.eigh(kmm)
        keep = eigenvalues > max(eigenvalues[-1]*1e-12, 1e-12)
        transform = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
        features = rbf(x, centers, lengthscale) @ transform
        system = features.T @ features
        system.flat[::len(system)+1] += n*regularization
        beta = cho_solve(cho_factor(system, lower=True), features.T @ y)
        return cls(centers, transform @ beta, lengthscale, clip, "nystrom_direct")

    def predict(self, x, batch_size=4096):
        out = np.empty(len(x))
        for start in range(0, len(x), batch_size):
            stop = min(start+batch_size, len(x))
            out[start:stop] = rbf(x[start:stop], self.centers, self.lengthscale) @ self.coefficients
        return np.clip(out, -self.clip, self.clip)

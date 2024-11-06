class RNN():
    def __init__(self, Q: int, P: int, tikhonov=False, lambda_=0.001) -> None:
        """
        Randomized Neural Network (RNN)
        
        Args:
            Q (int): The number of hidden neurons. This also specifies the dimension
                of the space where the feature are projected.
            P (int): The number of components in the input feature vector.
            tikhonov (bool, optional): Use Tikhonov's regularization.
            lambda_ (float, optional): Specifies the Tikhonov's regularization value.
        """
        self._Q = Q  # The hidden layer neuron count.
        self._P = P  # The input matrix rows count. (Number of attributes)

        self._lambda = lambda_  # Tikhonov's regularization parameter.
        self._tikhonov = tikhonov  # Use Tikhonov's regularization.

        # Hidden layer's output activation function.
        self._activation = torch.sigmoid

        # Generate the random weight matrix and the bias matrix.
        self._weights = self._setup_weight_matrix()

        # Does use Tikhonov's regularization?
        if tikhonov:
            self._eye = lambda_ * torch.eye(Q + 1, dtype=torch.float)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Fit the Randomized Neural Network (RNN).
        
        Args:
            X (torch.Tensor): The input feature matrix of dimensions (p, N).
            Y (torch.Tensor): The output feature matrix of dimensions (r, N).
            
        Notes:
            The second dimension of the `X` and `Y` matrices must be equal.
        """
        beta = None

        # Add the bias to X.
        X = torch.vstack([torch.ones((X.shape[1], )) * -1., X])

        # Hidden layer's output matrix.
        Z = self._activation(torch.mm(self._weights, X))

        # Add the bias to Z.
        Z = torch.vstack([torch.ones((Z.shape[1], )) * -1., Z])

        if self._tikhonov:
            # Calculates beta applying Tikhonov's regularization.
            beta = torch.mm(torch.mm(Y, Z.t()), torch.linalg.inv(
                torch.add(torch.mm(Z, Z.t()), self._eye)))
        else:
            # Calculates beta using least squares.
            beta = torch.linalg.lstsq(Z.t(), Y.t()).solution.t()

        self.beta = beta

    def _setup_weight_matrix(self) -> torch.Tensor:
        """
        Builds the Random Weight Matrix.
        
        The generation of the same random weight matrix ensures that the generated
        feature vectors are the same between multiple runs.
        
        Furthermore, this weight matrix is randomly generated using the Linear
        Congruent Generator (LCG), using the formula
        
            `V(n) = a * V(n - 1) % c`
            
        where a = L + 2, b = L + 3, c = L^2, and L = Q * (P + 1).
        
        Finally, the vector `V` is z-scored, and it is reshaped into the dimensions
        (Q, P + 1) transforming it into a random weight matrix.
        
        Returns:
            (torch.Tensor): The z-scored random weight matrix.
        """
        L = self._Q * (self._P + 1)

        V = torch.zeros(L, dtype=torch.float)
        V[0] = L + 1

        a = L + 2
        b = L + 3
        c = L * L

        for x in range(1, L):
            V[x] = (a * V[x-1] + b) % c

        # Always keep the zscore normalization for our LCG weights
        V = torch.divide(torch.subtract(V, torch.mean(V)), torch.std(V))

        return V.reshape((self._Q, self._P + 1))

    def _generate_images(self, img: np.ndarray) -> np.ndarray:
        X = torch.Tensor(img)
        # Add the bias to X.
        X = torch.vstack([torch.ones((X.shape[1], )) * -1., X])
        
        mean = torch.mean(X, dim=1, keepdim=True)
        std = torch.std(X, dim=1, keepdim=True)
        X = torch.divide(torch.subtract(X, mean), std + 1e-3)

        # Hidden layer's output matrix.
        Z = self._activation(torch.mm(self._weights, X))
        
        return Z.numpy()
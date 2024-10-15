"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.
class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        middle = self.layer1.forward(x).relu()
        end = self.layer2.forward(middle).relu()
        return self.layer3.forward(end).sigmoid()

class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.out_size = out_size
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)

    def forward(self, inputs):
        """Manually implement matrix multiplication by broadcasting inputs and weights for element-wise multiplication;
        with the equation of `x @ self.weights.value + self.bias.value`;
        Inputs x typically has a shape of (batch_size, input_size), whereas the weights self.weights
        typically has a shape of (input_size, output_size); Normally, matrix multiplication multiplies
        (batch_size, input_size) matrix by an (input_size, output_size) matrix, resulting in a
        (batch_size, output_size) output. However, we are manually implementing matrix multiplication
        (element-wise multiplication followed by summing), we need to align the shapes of x and self.weights
        so that broadcasting can occur correctly. For element-wise multiplication, we need to reshape using
        .view both x and weights to ensure that each element of x multiplies with the correct element in weights."""
        # Since MatMul is not implemented yet, use .view to change the shape of the tensor and any operators that are already defined
        # For inputs x reshaping, reshapes x from (batch_size, input_size) to (batch_size, input_size, 1)
        # so that it can broadcast correctly with the weights
        inputs_x_broadcasted = inputs.view(*inputs.shape, 1)  # (batch_size, input_size, 1)
        # For weights reshaping, reshapes the weights from (input_size, output_size) to (1, input_size, output_size)
        # to align with the reshaped input tensor
        weights_broadcasted = self.weights.value.view(1, *self.weights.value.shape)  # (1, input_size, output_size)

        # Element-wise multiplication for two tensors of same dimensions with (batch_size, input_size, output_size) for `weighted_inputs`
        # Applied for each batch independently, meaning that for each input feature (of size input_size), applying a different
        # weight to compute multiple outputs (of size output_size)
        weighted_inputs = inputs_x_broadcasted * weights_broadcasted

        # The sum along dim=1 reduces the input_size dimension, leaving with a tensor of shape (batch_size, output_size);
        # Sum over the input dimension (input_size) to get the linear output for each sample;
        # Reshape `linear_output` into (batch_size, output_size), ensuring the tensor is stored contiguously in memory,
        # which improves performance and ensures compatibility with other operations that require contiguous tensors.
        linear_output = weighted_inputs.sum(dim=1).contiguous().view(inputs.shape[0], self.out_size)

        # Broadcast the bias to match the output shape
        bias_broadcasted = self.bias.value.view(1, self.out_size)
        # Add bias to the linear output
        result = linear_output + bias_broadcasted

        return result


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)

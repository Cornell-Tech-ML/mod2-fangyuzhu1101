from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce placeholder"""
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            start: Initial value to start the reduction. Default is 0.0.
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
        -------
            :class:`TensorData` : new tensor

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.
def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError("Need to implement for Task 2.3")

        # Get the total number of elements in the output tensor by multiplying all dimensions
        num_elements_out_tensor = int(np.prod(out_shape))

        # Iterate over every index to initialize in the output tensor
        output_index = np.zeros_like(out_shape, dtype=np.int32)
        input_index = np.zeros_like(in_shape, dtype=np.int32)

        for flat_index in range(num_elements_out_tensor):
            # Convert the flat index to multidimensional indices for the output tensor
            to_index(flat_index, out_shape, output_index)

            # Broadcast the multidimensional index to the input tensor's shape
            broadcast_index(output_index, out_shape, in_shape, input_index)

            # Get the flat positions for in and out tensors of the current index in the storage arrays
            out_pos = index_to_position(output_index, out_strides)
            in_pos = index_to_position(input_index, in_strides)

            # Apply the function to the input and store result in the output array
            out[out_pos] = fn(in_storage[in_pos])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
    ----
        fn: function mapping two floats to float to apply

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError("Need to implement for Task 2.3")

        # Get the total number of elements in the output tensor by multiplying all dimensions
        num_elements_out_tensor = int(np.prod(out_shape))

        # Iterate over every index to initialize in the output tensor
        output_index = np.zeros_like(out_shape, dtype=np.int32)
        a_index = np.zeros_like(a_shape, dtype=np.int32)
        b_index = np.zeros_like(b_shape, dtype=np.int32)

        for flat_index in range(num_elements_out_tensor):
            # Convert the flat index to multidimensional indices for the output tensor
            to_index(flat_index, out_shape, output_index)

            # Broadcast the multidimensional index to the input tensors' shapes
            broadcast_index(output_index, out_shape, a_shape, a_index)
            broadcast_index(output_index, out_shape, b_shape, b_index)

            # Get the flat positions for a, b, and out tensors
            out_pos = index_to_position(output_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            # Apply the function to the inputs and store in the output
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError("Need to implement for Task 2.3")

        # Get the total number of elements in the output tensor by multiplying all dimensions
        num_elements_out_tensor = int(np.prod(out_shape))

        # Iterate over every index to initialize in the output tensor
        output_index = np.zeros_like(out_shape, dtype=np.int32)

        for flat_index in range(num_elements_out_tensor):
            # Convert the flat index to multidimensional indices for the output tensor
            to_index(flat_index, out_shape, output_index)

            # Initialize the result to the identity element (start value)
            reduction_result = 0.0
            # Iterate over the elements of the dimension to reduce
            for reduce_index in range(a_shape[reduce_dim]):
                input_index = np.copy(output_index)
                # Set the current index in the reduce dimension for the input tensor
                input_index[reduce_dim] = reduce_index

                # Convert the input_index to a flat position in the input storage array
                input_position = index_to_position(input_index, a_strides)

                # Apply the reduction function for the first element in the reduce dimension
                if reduce_index == 0:
                    reduction_result = a_storage[input_position]
                # Else, apply the reduction function for subsequent elements
                else:
                    reduction_result = fn(reduction_result, a_storage[input_position])

            # Store the reduction result in the output tensor
            output_pos = index_to_position(output_index, out_strides)
            out[output_pos] = reduction_result

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)

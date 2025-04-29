import numpy as np
import math
import torch
from typing import Optional, Union

from utils import vn_c_reshape

"""
This file contains code adapted from the PyTorch3D library:
https://github.com/facebookresearch/pytorch3d

Original PyTorch3D is Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the BSD license (see LICENSE file in the repository root for details).

If you use this code or PyTorch3D in your research, please cite:
Ravi et al., "Accelerating 3D Deep Learning with PyTorch3D", arXiv:2007.08501
"""

Device = Union[str, torch.device]

def rotation(batch_x1, rot, device):
    # batch_x1: (B, 1, L, C)
    time_length = batch_x1.size(2)
    batch_x1 = vn_c_reshape(batch_x1, time_length) # (B, L, C) -> (B, L, 3, C//3)

    batch_x1 = batch_x1.transpose(-2, -1) # (B, L, C//3, 3)

    trot = None

    if rot == "so3":
        trot = Rotate(batch_x1.shape[0], 'so3', device)
    elif rot == "z":
        trot = Rotate(batch_x1.shape[0], 'z', device)

    if trot is not None:
        batch_x1 = trot.transform_points(batch_x1)

    # back to original input shape 
    batch_x1 = batch_x1.flatten(start_dim=-2) # (B, L, C)
    batch_x1 = batch_x1.unsqueeze(1) # (B, 1, L ,C) - original shape

    return batch_x1

class Rotate:
  def __init__(self, n: int, method: str, device: Optional[Device], angle=None):
    """
    Args:
        n: Number of rotations in a batch to return.
        method: method of rotation, one of ['so3', 'x', 'y', 'z']
        device: Desired device of returned tensor. 
          Default: uses the current device for the default tensor type.
        angle: Euler angle in degrees
          Default: None - creates a size n tensor of random angles in radians [0, 2*pi] 

    """
    self.n = n
    self.method = method.lower()
    self.device = device

    if not angle:
      self.angle = torch.rand(self.n, device=device) * (360 / 180 * math.pi)
    else:
      self.angle = torch.full((self.n,), angle / 180 * math.pi, device=device) 

    if self.method == "so3":
      self.rotation = self.quaternion_to_matrix(self._random_quaternions())
    elif self.method in ["x", "y", "z"]:
      self.rotation = self._axis_angle_rotation(self.method, self.angle)
    else:
      msg = "Expected method to be one of ['so3', 'x', 'y', 'z']; got %s"
      raise ValueError(msg % method)

    self.rotation = self.rotation.transpose(1,2)
    # We assume the points on which this transformation will be applied
    # are row vectors. The rotation matrix returned from _axis_angle_rotation
    # is for transforming column vectors. Therefore we transpose this matrix.
    # rotation will always be of shape (N, 3, 3)

  def _random_quaternions(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
      """
      Generate random quaternions representing rotations,
      i.e. versors with nonnegative real part.


      Returns:
          Quaternions as tensor of shape (N, 4).
      """
      if isinstance(self.device, str):
          self.device = torch.device(self.device)
      o = torch.randn((self.n, 4), dtype=dtype, device=self.device)
      s = (o * o).sum(1)
      o = o / self._copysign(torch.sqrt(s), o[:, 0])[:, None]
      return o
      
  def _axis_angle_rotation(self, method: str, angle: torch.Tensor) -> torch.Tensor:
      """
      Return the rotation matrices for one of the rotations about an axis
      of which Euler angles describe, for each value of the angle given.

      Args:
          method: Axis label "x" or "y" or "z".
          angle: any shape tensor of Euler angles in radians

      Returns:
          Rotation matrices as tensor of shape (..., 3, 3).
      """
      cos = torch.cos(angle)
      sin = torch.sin(angle)
      one = torch.ones_like(angle)
      zero = torch.zeros_like(angle)

      if self.method == "x":
          R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
      elif self.method == "y":
          R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
      elif self.method == "z":
          R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
      else:
          raise ValueError("letter must be either x, y or z.")

      return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))
    
  def transform_points(self, points, eps: Optional[float] = None) -> torch.Tensor:
      """
      Use this transform to transform a set of 3D points. Assumes row major
      ordering of the input points.

      Args:
          points: Tensor of shape (P, 3) or (N, P, 3)
          eps: If eps!=None, the argument is used to clamp the
              last coordinate before performing the final division.
              The clamping corresponds to:
              last_coord := (last_coord.sign() + (last_coord==0)) *
              torch.clamp(last_coord.abs(), eps),
              i.e. the last coordinates that are exactly 0 will
              be clamped to +eps.

      Returns:
          points_out: points of shape (N, P, 3) or (P, 3) depending
          on the dimensions of the transform
      """
      points_batch = points.clone()
    #   if points_batch.dim() == 2:
    #       points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
    #   if points_batch.dim() != 3:
    #       msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
    #       raise ValueError(msg % repr(points.shape))

      points_out = self._broadcast_bmm(points_batch, self.rotation) # (N, P, 3) @ (N, 3, 3)

      return points_out

  def _copysign(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)
    
  def quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

  def _broadcast_bmm(self, a, b) -> torch.Tensor:
    """
    Batch multiply two matrices and broadcast if necessary.

    Args:
        a: torch tensor of shape (P, K) or (M, P, K)
        b: torch tensor of shape (N, K, K)

    Returns:
        a and b broadcast multiplied. The output batch dimension is max(N, M).

    To broadcast transforms across a batch dimension if M != N then
    expect that either M = 1 or N = 1. The tensor with batch dimension 1 is
    expanded to have shape N or M.
    """
    if a.dim() == 2:
        a = a[None]
    if len(a) != len(b):
        if not ((len(a) == 1) or (len(b) == 1)):
            msg = "Expected batch dim for bmm to be equal or 1; got %r, %r"
            raise ValueError(msg % (a.shape, b.shape))
        if len(a) == 1:
            a = a.expand(len(b), -1, -1)
        if len(b) == 1:
            b = b.expand(len(a), -1, -1)
    return a.bmm(b)

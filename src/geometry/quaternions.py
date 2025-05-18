import torch
import torch.nn.functional as F
from src.geometry.vector import normalize_vector
import numpy as np
def rotation_6d_to_matrix_no_normalized(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    # b1 = F.normalize(a1, dim=-1)
    # b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    # b2 = F.normalize(b2, dim=-1)
    # b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((a1, a2), dim=-2)

def rotation6d_multiply(r1,r2):
    r1 = rotation_6d_to_matrix(r1)
    r2 = rotation_6d_to_matrix(r2)
    return matrix_to_rotation_6d(torch.matmul(r1,r2))
def rotation6d_apply(r,p):
    #p :...3
    r = rotation_6d_to_matrix(r)
    p = p.unsqueeze(-1)#...3,1
    return torch.matmul(r,p).squeeze(-1)
def rotation6d_inverse(r):
    r = rotation_6d_to_matrix(r)
    inv_r = torch.transpose(r,-2,-1)
    return matrix_to_rotation_6d(inv_r)
def quat_to_or6D(quat):

    assert(quat.shape[-1]==4)
    return matrix_to_rotation_6d(quaternion_to_matrix(quat))
def or6d_to_quat(mat):
    assert (mat.shape[-1] == 6)
    return matrix_to_quaternion(rotation_6d_to_matrix(mat))
def normalized_or6d(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    return torch.cat((b1, b2), dim=-1)
#移除四元数不连续的问题
def remove_quat_discontinuities(rotations):
    rotations = rotations.clone()
    rots_inv = -rotations
    for i in range(1, rotations.shape[1]):
        replace_mask = torch.sum(rotations[:, i-1:i, ...] * rotations[:, i:i+1, ...], 
                                 dim=-1, keepdim=True) < \
                       torch.sum(rotations[:, i-1:i, ...] * rots_inv[:, i:i+1, ...], 
                                 dim=-1, keepdim=True)
        replace_mask = replace_mask.squeeze(1).type_as(rotations)
        rotations[:, i, ...] = replace_mask * rots_inv[:, i, ...] + (1.0 - replace_mask) * rotations[:, i, ...]
    return rotations

def from_to_1_0_0(v_from):
    '''v_from: B,3'''
    v_to_unit = torch.tensor(np.array([[1, 0, 0]]), dtype=v_from.dtype, device=v_from.device).expand(v_from.shape)
    y = torch.tensor(np.array([[0,1,0]]),dtype=v_from.dtype,device=v_from.device).expand(v_from.shape)
    z_to_unit = torch.tensor(np.array([[0,0,1]]),dtype=v_from.dtype,device=v_from.device).expand(v_from.shape)

    v_from_unit = normalize_vector(v_from)
    z_from_unit = normalize_vector(torch.cross(v_from_unit,y,dim=-1))
    to_transform = torch.stack([v_to_unit,y,z_to_unit],dim=-1)
    from_transform = torch.stack([v_from_unit,y,z_from_unit],dim=-1)
    shape = to_transform.shape
    to_transform = to_transform.view(-1,3,3)
    from_transform = from_transform.view(-1,3,3)
    r = torch.matmul(to_transform,from_transform.transpose(1,2)).view(shape)
    rq = matrix_to_quaternion(r)

    # w = (v_from_unit * v_to_unit).sum(dim=1) + 1
    # '''can't cross if two directions are exactly inverse'''
    # xyz = torch.cross(v_from_unit, v_to_unit, dim=1)
    # '''if exactly inverse, the rotation should be (0,0,1,0), around yaxis 180'''
    # xyz[...,1] = torch.where(w==0,torch.tensor([1],dtype=v_from.dtype,device=xyz.device),xyz[...,1])
    # q = torch.cat([w.unsqueeze(1), xyz], dim=1)
    return rq


# returns quaternion so that v_from rotated by this quaternion equals v_to
# v_... are vectors of size (..., 3)
# returns quaternion in w, x, y, z order, of size (..., 4)
# note: such a rotation is not unique, there is an infinite number of solutions
# this implementation returns the shortest arc
def from_to_quaternion(v_from, v_to):
    v_from_unit = normalize_vector(v_from)
    v_to_unit = normalize_vector(v_to)

    w = (v_from_unit * v_to_unit).sum(dim=-1)+1
    '''can't cross if two directions are exactly inverse'''
    xyz = torch.cross(v_from_unit, v_to_unit, dim=-1)


    q = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
    return normalize_vector(q)

def quat_inv(quat):
    return quaternion_invert(quat)
def quat_mul(quat0,quat1):
    return quaternion_multiply(quat0,quat1)
def quat_mul_vec(quat,vec):
    return quaternion_apply(quat,vec)
def slerp(q0, q1, t):
    """
    Spherical Linear Interpolation of quaternions
    https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm
    :param q0: Start quats (w, x, y, z) : shape = (B, J, 4)
    :param q1: End quats (w, x, y, z) : shape = (B, J, 4)
    :param t:  Step (in [0, 1]) : shape = (B, J, 1)
    :return: Interpolated quat (w, x, y, z) : shape = (B, J, 4)
    """
  #  q0 = q0.unsqueeze(1)
  #  q1 = q1.unsqueeze(1)
    
    # Dot product
    q = q0*q1
    cos_half_theta = torch.sum(q, dim=-1, keepdim=True)
   # t = t.view(1,-1,1,1)
    # Make sure we take the shortest path :
    q1_antipodal = -q1
    q1 = torch.where(cos_half_theta < 0, q1_antipodal, q1)
    cos_half_theta = torch.where(cos_half_theta < 0,-cos_half_theta,cos_half_theta)
    half_theta = torch.acos(cos_half_theta)
    # torch.sin must be safer here
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)
    ratio_a = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = torch.sin(t * half_theta) / sin_half_theta
    
    qt = ratio_a * q0 + ratio_b * q1    
    # If the angle was constant, prevent nans by picking the original quat:
    qt = torch.where(torch.abs(cos_half_theta) >= 1.0-1e-8, q0, qt)
    return qt

def quaternion_to_matrix(q):
    """Converte um quaternion (w, x, y, z) para matriz de rotação 3x3"""
    w, x, y, z = q.unbind(-1)
    B = q.shape[:-1]
    R = torch.empty(B + (3, 3), dtype=q.dtype, device=q.device)

    R[..., 0, 0] = 1 - 2*y*y - 2*z*z
    R[..., 0, 1] = 2*x*y - 2*z*w
    R[..., 0, 2] = 2*x*z + 2*y*w

    R[..., 1, 0] = 2*x*y + 2*z*w
    R[..., 1, 1] = 1 - 2*x*x - 2*z*z
    R[..., 1, 2] = 2*y*z - 2*x*w

    R[..., 2, 0] = 2*x*z - 2*y*w
    R[..., 2, 1] = 2*y*z + 2*x*w
    R[..., 2, 2] = 1 - 2*x*x - 2*y*y
    return R

def matrix_to_quaternion(M):
    """Converte matriz de rotação 3x3 para quaternion (w, x, y, z)"""
    m = M
    B = m.shape[:-2]
    q = torch.empty(B + (4,), dtype=m.dtype, device=m.device)

    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cond = t > 0
    not_cond = ~cond

    t_sqrt = torch.sqrt(1.0 + t[cond]) * 2
    q[cond, 0] = 0.25 * t_sqrt
    q[cond, 1] = (m[cond, 2, 1] - m[cond, 1, 2]) / t_sqrt
    q[cond, 2] = (m[cond, 0, 2] - m[cond, 2, 0]) / t_sqrt
    q[cond, 3] = (m[cond, 1, 0] - m[cond, 0, 1]) / t_sqrt

    # fallback para casos onde t <= 0
    q[not_cond] = torch.tensor([1, 0, 0, 0], dtype=m.dtype, device=m.device)

    return q

def rotation_6d_to_matrix(d6):
    """Converte vetor 6D para matriz de rotação (Zhou et al. 2019)"""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_rotation_6d(matrix):
    """Converte matriz de rotação para vetor 6D"""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)

def quaternion_multiply(q, r):
    """ Multiplica dois quaternions: q * r """
    assert q.shape[-1] == 4 and r.shape[-1] == 4
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)
quaternion_to_matrix
def quaternion_apply(q, v):
    """ Aplica o quaternion `q` a um vetor 3D `v` """
    q_conj = q.clone()
    q_conj[..., 1:] = -q_conj[..., 1:]
    v_as_quat = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return quaternion_multiply(
        quaternion_multiply(q, v_as_quat),
        q_conj
    )[..., 1:]

def quaternion_invert(q):
    """ Inverte o quaternion `q` """
    q_conj = q.clone()
    q_conj[..., 1:] *= -1
    norm_squared = (q ** 2).sum(dim=-1, keepdim=True)
    return q_conj / (norm_squared + 1e-8)

def axis_angle_to_quaternion(axis_angle):
    """ Converte eixo e ângulo para quaternion """
    angles = axis_angle.norm(dim=-1, keepdim=True)
    axis = axis_angle / (angles + 1e-8)
    half = 0.5 * angles
    sin_half = torch.sin(half)
    cos_half = torch.cos(half)
    return torch.cat([cos_half, axis * sin_half], dim=-1)
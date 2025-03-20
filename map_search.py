import torch
from cryodrgn.lattice import Lattice
from cryodrgn.pose_search import rot_2d, interpolate
from cryodrgn import fft
import pickle
import torch.nn.functional as F


def downsample_vol(map, res, target_res=5, target_size=128):
    """
    Downsample a DxDxD image to a target resolution and pad to target size.

    Args:
        map: Tensor of shape D x D x D, input volume
        res: Current resolution of the input volume in A/pix
        target_res: Target resolution in A/pix (default is 5 A/pix)
        target_size: Target size for padding (default is 128)

    Returns:
        Tensor of shape target_size x target_size x target_size, downsampled and padded volume
    """
    if res < target_res:
        scale_factor = res / target_res
        D = map.shape[0]
        new_D = int(D * scale_factor)
        grid = torch.meshgrid([torch.linspace(-1, 1, new_D) for _ in range(3)], indexing='ij')
        grid = torch.stack(grid, dim=-1).unsqueeze(0).to(map.device)
        map = F.grid_sample(map.unsqueeze(0).unsqueeze(0), grid, align_corners=True).squeeze()

    pad_width = sum([((target_size - s) // 2, (target_size - s + 1) // 2) for s in map.shape], ())
    map = F.pad(map, pad_width, mode='constant', value=0)

    return map

def translate_maps(maps, trans, input_hartley=True, output_hartley=True):
    """
    Translate 3D maps in Hartley/real space by phase shifting

    Args:
        maps: Tensor of shape N x D x D x D, maps in real or Hartley space
        trans: Tensor of translations to apply, T x 3 or N x T x 3
        input_hartley: Bool indicating if input maps are in Hartley space
        output_hartley: Bool indicating if output should be in Hartley space

    Returns:
        Tensor of translated maps with shape N x T x D x D x D, where T is number of translations.
        Output is in Hartley or real space based on output_hartley parameter.
    """

    if not input_hartley:
        ht = fft.ht3_center(maps)
        ht = fft.symmetrize_ht(ht)
    else:
        ht = maps

    if len(trans.shape) == 2:
        trans = trans.unsqueeze(0)

    # Apply phase shift for translation in Hartley space
    grid = torch.fft.fftfreq(ht.shape[1], d=1/ht.shape[1], device=ht.device)
    grid = torch.stack(torch.meshgrid(grid, grid, grid, indexing='ij'), dim=-1)
    phase_shift = torch.exp(-2j * np.pi * (grid.unsqueeze(0).unsqueeze(0) * trans.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)).sum(dim=-1))
    translated_ht = ht.unsqueeze(1) * phase_shift

    if not output_hartley:
        translated_maps = fft.iht3_center(translated_ht)
    else:
        translated_maps = translated_ht

    return translated_maps

def generate_rotated_slices(D, rotation_matrices):
    """
    Generate a N x D x D x 3 slice array, where each slice is a D x D x 3 meshgrid of 3D coordinates
    with the center at 0. Each slice is rotated by the corresponding rotation matrix.

    Args:
        D: Dimension of the slice
        rotation_matrices: Tensor of shape N x 3 x 3, rotation matrices

    Returns:
        Tensor of shape N x D x D x 3, rotated slices
    """
    N = rotation_matrices.shape[0]
    # Create base slice with extent DxD in the xy plane centered at the origin
    x = torch.linspace(-D//2, D//2, D)
    y = torch.linspace(-D//2, D//2, D)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    base_slice = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1).to(rotation_matrices.device)  # D x D x 3

    # Apply rotation matrices to the base slice
    rotated_slices = torch.zeros((N, D, D, 3), device=rotation_matrices.device)
    for i in range(N):
        rotated_slices[i] = torch.einsum('ij,xyj->xyi', rotation_matrices[i], base_slice)

    return rotated_slices

def optimize_rot_trans(ref_maps, query_maps, query_rotation_matrices, ref_rotation_offsets, translation_vectors):
    """
    Optimize rotation and translation for query and reference maps.

    Args:
        query_maps: Tensor of shape N x D x D x D, query maps
        ref_maps: Tensor of shape M x D x D x D, reference maps 
        query_rotation_matrices: Tensor of shape R_q x 3 x 3, rotation matrices for query maps
        ref_rotation_offsets: Tensor of shape R_r x 3 x 3, rotation offsets for reference maps
        translation_vectors: Tensor of shape T x 3, translation vectors

    Returns:
        correlations: Tensor of shape N x M x T x R_r, normalized correlation values
    """
    N, D, _, _ = query_maps.shape  # N x D x D x D
    M, _, _, _ = ref_maps.shape    # M x D x D x D
    R_q, _, _ = query_rotation_matrices.shape  # R_q x 3 x 3
    R_r, _, _ = ref_rotation_offsets.shape     # R_r x 3 x 3
    T, _ = translation_vectors.shape  # T x 3

    # Generate rotated slices for query maps
    rotated_slices_query = generate_rotated_slices(D, query_rotation_matrices)  # R_q x D x D x 3

    # Compose ref rotation matrices: R_q x R_r x 3 x 3
    ref_rotation_matrices = torch.einsum('ijkl, ijlr->ijkr', 
                                       query_rotation_matrices.unsqueeze(1),  # R_q x 1 x 3 x 3
                                       ref_rotation_offsets.unsqueeze(0))     # 1 x R_r x 3 x 3

    # Generate rotated slices for reference maps: (R_q*R_r) x D x D x 3
    rotated_slices_ref = generate_rotated_slices(D, ref_rotation_matrices.reshape(-1, 3, 3))

    # Prepare grid for grid_sample
    grid_query = rotated_slices_query[...,:2]  # R_q x D x D x 2
    grid_query = grid_query.unsqueeze(0).expand(N, -1, -1, -1, -1)  # N x R_q x D x D x 2
    
    grid_ref = rotated_slices_ref[...,:2].view(R_q, R_r, D, D, 2)  # R_q x R_r x D x D x 2
    grid_ref = grid_ref.unsqueeze(0).expand(M, -1, -1, -1, -1, -1)  # M x R_q x R_r x D x D x 2

    # Translate query maps: N x T x D x D x D
    translated_query_maps = query_maps.unsqueeze(1)  # N x 1 x D x D x D

    # Extract central slices from translated query maps using grid_query
    translated_query_maps = translated_query_maps.reshape(N*T, 1, D, D, D)
    grid_query = grid_query.repeat(T, 1, 1, 1, 1)  # (N*T) x R_q x D x D x 2
    translated_rotated_query = F.grid_sample(translated_query_maps, 
                                           grid_query.reshape(-1, D, D, 2),
                                           align_corners=True)  # (N*T) x 1 x D x D
    translated_rotated_query = translated_rotated_query.view(N, T, R_q, D, D)

    # Extract central slices from reference maps using grid_ref
    ref_maps = ref_maps.unsqueeze(1).unsqueeze(1)  # M x 1 x 1 x D x D x D
    sliced_ref = F.grid_sample(ref_maps.reshape(M, 1, D, D, D),
                              grid_ref.reshape(M*R_q*R_r, D, D, 2),
                              align_corners=True)  # M x 1 x D x D
    sliced_ref = sliced_ref.view(M, R_q, R_r, D, D)

    # Compute correlations
    translated_rotated_query = translated_rotated_query.unsqueeze(1)  # N x 1 x T x R_q x D x D
    sliced_ref = sliced_ref.unsqueeze(0).unsqueeze(2)  # 1 x M x 1 x R_q x R_r x D x D

    # Normalize inputs
    query_std = translated_rotated_query.std(dim=(-2,-1), keepdim=True)  # N x 1 x T x R_q x 1 x 1
    ref_std = sliced_ref.std(dim=(-2,-1), keepdim=True)  # 1 x M x 1 x R_q x R_r x 1 x 1
    
    query_norm = (translated_rotated_query - translated_rotated_query.mean(dim=(-2,-1), keepdim=True)) / query_std
    ref_norm = (sliced_ref - sliced_ref.mean(dim=(-2,-1), keepdim=True)) / ref_std

    # Compute correlation
    corr = (query_norm * ref_norm).sum(dim=(-2,-1))  # N x M x T x R_q x R_r
    corr = corr.mean(dim=3)  # Average over R_q: N x M x T x R_r

    return corr



def optimize_theta_trans_chunked(ref_maps, query_maps, trans, rot, chunk_size=100):
    N = query_maps.shape[0]
    M = ref_maps.shape[0]
    correlations_list = []

    for i in range(0, M, chunk_size):
        ref_chunk = ref_maps[i:i + chunk_size]
        correlations_chunk = optimize_rot_trans(ref_chunk, query_maps, trans, rot)
        correlations_list.append(correlations_chunk)

    correlations = torch.cat(correlations_list, dim=1)
    return correlations

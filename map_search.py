import torch
from cryodrgn.lattice import Lattice
from cryodrgn.pose_search import rot_2d, interpolate
from cryodrgn import fft
import pickle

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
    base_slice = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1)  # D x D x 3

    # Apply rotation matrices to the base slice
    rotated_slices = torch.zeros((N, D, D, 3))
    for i in range(N):
        rotated_slices[i] = torch.einsum('ij,xyj->xyi', rotation_matrices[i], base_slice)

    return rotated_slices

def optimize_rot_trans(ref_maps, query_maps, query_rotation_matrices, ref_rotation_offsets, translation_vectors):
    """
    Optimize rotation and translation for query and reference maps.

    Args:
        query_maps: Tensor of shape N x D x D, query maps
        ref_maps: Tensor of shape M x D x D, reference maps
        query_rotation_matrices: Tensor of shape R_q x 3 x 3, rotation matrices for query maps
        ref_rotation_offsets: Tensor of shape R_r x 3 x 3, rotation offsets for reference maps
        translation_vectors: Tensor of shape T x 2, translation vectors

    Returns:
        correlations: Tensor of shape N x M x T x R_r, normalized correlation values
    """
    N, D, _ = query_maps.shape
    M, _, _ = ref_maps.shape
    R_q, _, _ = query_rotation_matrices.shape
    R_r, _, _ = ref_rotation_offsets.shape
    T, _ = translation_vectors.shape

    # Generate rotated slices for query maps
    rotated_slices_query = generate_rotated_slices(D, query_rotation_matrices)

    # Compose ref rotation matrices
    ref_rotation_matrices = torch.einsum('ijk,klm->ijlm', query_rotation_matrices.unsqueeze(1), ref_rotation_offsets.unsqueeze(0))

    # Generate rotated slices for reference maps
    rotated_slices_ref = generate_rotated_slices(D, ref_rotation_matrices.view(-1, 3, 3))

    # Prepare grid for grid_sample
    grid_query = rotated_slices_query[:, :, :, :2].view(R_q, D, D, 2).unsqueeze(0).repeat(N, 1, 1, 1, 1).to(query_maps.device)
    grid_ref = rotated_slices_ref[:, :, :, :2].view(R_q * R_r, D, D, 2).unsqueeze(0).repeat(M, 1, 1, 1, 1).to(ref_maps.device)

    # Translate query images
    query_maps_expanded = query_maps.unsqueeze(1).repeat(1, T, 1, 1).view(N * T, 1, D, D)
    translation_grid = translation_vectors.view(1, T, 1, 1, 2).repeat(N, 1, D, D, 1).to(query_maps.device)
    translated_query_images = torch.nn.functional.grid_sample(query_maps_expanded, translation_grid, align_corners=True)
    translated_query_images = translated_query_images.view(N, T, D, D)

    # Generate translated and rotated query images
    translated_query_images_expanded = translated_query_images.unsqueeze(2).repeat(1, 1, R_q, 1, 1).view(N * T * R_q, 1, D, D)
    grid_query = grid_query.view(1, R_q, D, D, 2).repeat(N * T, 1, 1, 1, 1).view(N * T * R_q, D, D, 2)
    translated_rotated_query_images = torch.nn.functional.grid_sample(translated_query_images_expanded, grid_query, align_corners=True)
    translated_rotated_query_images = translated_rotated_query_images.view(N, T, R_q, D, D)

    # Generate sliced reference images
    ref_maps_expanded = ref_maps.unsqueeze(1).repeat(1, R_q * R_r, 1, 1).view(M * R_q * R_r, 1, D, D)
    grid_ref = grid_ref.view(M * R_q * R_r, D, D, 2)
    sliced_ref_images = torch.nn.functional.grid_sample(ref_maps_expanded, grid_ref, align_corners=True)
    sliced_ref_images = sliced_ref_images.view(M, R_q, R_r, D, D)

    # Correlate sliced reference and query images
    translated_rotated_query_images = translated_rotated_query_images.unsqueeze(2)  # N x T x 1 x R_q x D x D
    sliced_ref_images = sliced_ref_images.unsqueeze(0).unsqueeze(0)  # 1 x 1 x M x R_q x R_r x D x D

    # Multiply and sum over D dimensions
    product = translated_rotated_query_images * sliced_ref_images
    sum_product = product.sum(dim=(4, 5))  # N x T x M x R_q x R_r

    # Normalize by standard deviation over D dimensions
    query_std = translated_rotated_query_images.std(dim=(3, 4), keepdim=True)
    ref_std = sliced_ref_images.std(dim=(4, 5), keepdim=True)
    normalized_corr = sum_product / (query_std * ref_std).squeeze()

    # Mean correlation over R_q dimensions
    mean_corr = normalized_corr.mean(dim=3)  # N x T x M x R_r

    # Permute axes to N x M x T x R_r
    correlations = mean_corr.permute(0, 2, 1, 3)

    return correlations



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

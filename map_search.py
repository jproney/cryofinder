import torch
import numpy as np
import torch.nn.functional as F
from cryodrgn.shift_grid import grid_1d


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


def translate_ht3(img, t, coords=None):
    """
    Translate an image by phase shifting its Hartley transform

    Inputs:
        img: HT of image (B x D x D x D)
        t: shift in pixels (B x T x 3)
        coords: N x 3

    Returns:
        Shifted images (B x T x img_dims)

    img must be 1D unraveled image, symmetric around DC component
    """

    B, D, _, _ = img.shape
    T, _ = t.shape
    if coords is None:
        # Create 3D meshgrid of size D
        x = torch.linspace(-0.5, 0.5, D, device=img.device)
        y = torch.linspace(-0.5, 0.5, D, device=img.device)
        z = torch.linspace(-0.5, 0.5, D, device=img.device)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    # H'(k) = cos(2*pi*k*t0)H(k) + sin(2*pi*k*t0)H(-k)
    img = img.view((img.shape[0], 1, -1)) # Bx1xN
    t = t.unsqueeze(-1)  # BxTx3x1 to be able to do bmm
    tfilt = coords @ t * 2 * np.pi  # BxTxNx1
    tfilt = tfilt.squeeze(-1)  # BxTxN
    c = torch.cos(tfilt)  # BxTxN
    s = torch.sin(tfilt)  # BxTxN

    return (c * img + s * img[:, :, torch.arange(len(coords) - 1, -1, -1)]).view((B,T,D,D,D))


def symmetrize_ht3(ht: torch.Tensor) -> torch.Tensor:
    if ht.ndim == 3:
        ht = ht[np.newaxis, ...]
    assert ht.ndim == 4
    n = ht.shape[0]

    D = ht.shape[-1]
    sym_ht = torch.empty((n, D + 1, D + 1, D + 1), dtype=ht.dtype, device=ht.device)
    sym_ht[:, 0:-1, 0:-1, 0:-1] = ht

    assert D % 2 == 0
    sym_ht[:, -1, -1, :] = sym_ht[:, 0, 0, :] 
    sym_ht[:, -1,  :,-1] = sym_ht[:, 0, :, 0] 
    sym_ht[:,  :, -1,-1] = sym_ht[:, :, 0, 0] 

    sym_ht[:, -1, -1,-1] = sym_ht[:, 0, 0, 0] 


    return sym_ht

def grid_3d(
    resol: int, extent: int, ngrid: int, xshift: int = 0, yshift: int = 0, zshift: int = 0
) -> np.ndarray:
    x = grid_1d(resol, extent, ngrid, shift=xshift)
    y = grid_1d(resol, extent, ngrid, shift=yshift)
    z = grid_1d(resol, extent, ngrid, shift=zshift)

    # convention: x is fast dim, y is slow dim
    grid = np.stack(np.meshgrid(x, y, z), -1)
    return grid.reshape(-1, 3)

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
    x = torch.linspace(-1, 1, D)
    y = torch.linspace(-1, 1, D)
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
    T = translation_vectors.shape[0] if translation_vectors is not None else 1 # T x 3 or None

    # Generate rotated slices for query maps
    rotated_slices_query = generate_rotated_slices(D, query_rotation_matrices)  # R_q x D x D x 3

    # Compose ref rotation matrices: R_q x R_r x 3 x 3
    ref_rotation_matrices = torch.einsum('ijkl, ijlr->ijkr', 
                                       ref_rotation_offsets.unsqueeze(0),
                                       query_rotation_matrices.unsqueeze(1)  # R_q x 1 x 3 x 3
                                       )     # 1 x R_r x 3 x 3

    # Generate rotated slices for reference maps: (R_q*R_r) x D x D x 3
    rotated_slices_ref = generate_rotated_slices(D, ref_rotation_matrices.reshape(-1, 3, 3))

    # Prepare grid for grid_sample
    grid_query = rotated_slices_query.unsqueeze(0).expand(N, -1, -1, -1, -1)  # N x R_q x D x D x 3

    
    grid_ref = rotated_slices_ref.unsqueeze(0).expand(M, -1, -1, -1, -1)  # M x (R_q*R_r) x D x D x 3

    # Translate query maps: N x T x D x D x D
    if translation_vectors is None:
        translated_query_maps = query_maps.unsqueeze(1)
    else:
        translated_query_maps = translate_ht3(query_maps, translation_vectors)

    # Extract central slices from translated query maps using grid_query
    translated_rotated_query = F.grid_sample(translated_query_maps, 
                                           grid_query,
                                           align_corners=True)  # N x T x R_q x D x D
    translated_rotated_query = translated_rotated_query.view(N, T, R_q, D, D)

    # Extract central slices from reference maps using grid_ref
    ref_maps = ref_maps.unsqueeze(1)  # M x 1 x D x D x D
    sliced_ref = F.grid_sample(ref_maps,
                              grid_ref,
                              align_corners=True)  # M x 1 x R_q*R_r x D x D
    sliced_ref = sliced_ref.view(M, R_q, R_r, D, D)

    del grid_ref, grid_query

    # Compute correlations
    translated_rotated_query = translated_rotated_query.unsqueeze(1).unsqueeze(4)  # N x 1 x T x R_q x 1 x D x D
    sliced_ref = sliced_ref.unsqueeze(0).unsqueeze(2)  # 1 x M x 1 x R_q x R_r x D x D

    # Normalize inputs
    query_std = torch.sqrt(translated_rotated_query.sum(dim=(-2,-1), keepdim=True))  # N x 1 x T x R_q x 1 x 1
    ref_std = torch.sqrt(sliced_ref.sum(dim=(-2,-1), keepdim=True))  # 1 x M x 1 x R_q x R_r x 1 x 1
    

    corr = ((translated_rotated_query * sliced_ref).sum(dim=(-2,-1), keepdim=True) / query_std / ref_std).mean(dim=(3,-2,-1)) # N x M x T x R_r
    br_corr, bestrots = corr.max(dim=-1)
    _, besttrans = torch.max(br_corr, dim=-1)
    bestrot = bestrots.gather(-1, besttrans.unsqueeze(-1))
    return corr, translated_rotated_query.view(N, T, R_q, D, D)[torch.arange(N).unsqueeze(1), besttrans], sliced_ref.view(M, R_q, R_r, D, D)[torch.arange(M).view([-1,1,1]), torch.arange(R_q).view([1,-1,1]), bestrot]



def optimize_theta_trans_chunked(ref_maps, query_maps, trans, rot, chunk_size=100):
    N = query_maps.shape[0]
    M = ref_maps.shape[0]
    correlations_list = []

    for i in range(0, M, chunk_size):
        ref_chunk = ref_maps[i:i + chunk_size]
        correlations_chunk, _, _ = optimize_rot_trans(ref_chunk, query_maps, trans, rot)
        correlations_list.append(correlations_chunk.cpu())

    correlations = torch.cat(correlations_list, dim=1)
    return correlations

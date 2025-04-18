import torch
import numpy as np
import torch.nn.functional as F
from cryodrgn.shift_grid import grid_1d
import pickle
from cryodrgn.fft import htn_center, ihtn_center



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


def translate_ht3(img, t, coords=None, input_hartley=True, output_hartley=True):
    """
    Translate an image by phase shifting its Hartley transform

    Inputs:
        img: HT of image (B x D x D x D)
        t: shift in pixels (B x T x 3)
        coords: N x 3

    Returns:
        Shifted images (B x T x img_dims)
    """

    if not input_hartley:
        img = symmetrize_ht3(torch.stack([htn_center(im) for im in img]))


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

    trans_img =  (c * img + s * img[:, :, torch.arange(len(coords) - 1, -1, -1)]).view((B,T,D,D,D))

    if not output_hartley:
        trans_img = torch.stack([torch.stack([ihtn_center(im) for im in batch]) for batch in trans_img])

    return trans_img

def symmetrize_ht3(ht: torch.Tensor) -> torch.Tensor:
    """
    Symmetrize a 3D Hartley transform by adding the required redundant planes.
    
    Args:
        ht (torch.Tensor): Input 3D Hartley transform of shape (D,D,D) or (B,D,D,D)
                          where D is the dimension and B is the batch size.
                          The input is assumed to be of size D along each dimension.
    
    Returns:
        torch.Tensor: Symmetrized Hartley transform of shape (B,D+1,D+1,D+1).
                     The output has size D+1 along each dimension to include
                     the redundant planes required for proper symmetry.
    """
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
    """
    Generate a 3D grid of coordinates.
    
    Args:
        resol (int): Resolution/spacing between grid points
        extent (int): Total extent/size of the grid
        ngrid (int): Number of grid points along each dimension
        xshift (int, optional): Shift of grid in x direction. Defaults to 0.
        yshift (int, optional): Shift of grid in y direction. Defaults to 0. 
        zshift (int, optional): Shift of grid in z direction. Defaults to 0.

    Returns:
        np.ndarray: Array of shape (ngrid^3, 3) containing 3D coordinates of grid points
    """
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

def optimize_rot_trans(ref_maps, query_maps, query_rotation_matrices, ref_rotation_offsets, translation_vectors, input_hartley=True, hartley_corr=True):
    """
    Optimize rotation and translation for query and reference maps.

    Args:
        ref_maps: Tensor of shape M x D x D x D, reference maps 
        query_maps: Tensor of shape N x D x D x D, query maps
        query_rotation_matrices: Tensor of shape R_q x 3 x 3, rotation matrices for query maps
        ref_rotation_offsets: Tensor of shape R_r x 3 x 3, rotation offsets for reference maps
        translation_vectors: Tensor of shape T x 3, translation vectors

    Returns:
        best_indices: Tensor of shape N x 3 containing indices of best matches for each query:
            - Column 0: Index of best reference map
            - Column 1: Index of best translation
            - Column 2: Index of best rotation
        bestcorr: Tensor of shape N containing correlation values for best matches
        corr: Tensor of shape N x M x T x R_r, normalized correlation values between query and reference slice stacks
        translated_rotated_query: Tensor of shape N x R_q x D x D, extracted slices from optimally translated query
        sliced_ref: Tensor of shape N x R_q x D x D, extracted slices from the optimally rotated, best-matching ereference for each query
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
        if not input_hartley and hartley_corr:
            query_maps = symmetrize_ht3(torch.stack([htn_center(qm) for qm in query_maps]))
        translated_query_maps = query_maps.unsqueeze(1)
    else:
        translated_query_maps = translate_ht3(query_maps, translation_vectors, input_hartley=input_hartley, output_hartley=hartley_corr)

    # Extract central slices from translated query maps using grid_query
    translated_rotated_query = F.grid_sample(translated_query_maps, 
                                           grid_query,
                                           align_corners=True)  # N x T x R_q x D x D
    translated_rotated_query = translated_rotated_query.view(N, T, R_q, D, D)

    # Extract central slices from reference maps using grid_ref
    if not input_hartley and hartley_corr:
        ref_maps = symmetrize_ht3(torch.stack([htn_center(rm) for rm in ref_maps]))

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
    query_mean = translated_rotated_query.mean(dim=(3,-2,-1), keepdim=True)  # N x 1 x T x R_q x 1 x 1
    ref_mean = sliced_ref.mean(dim=(3,-2,-1), keepdim=True)# 1 x M x 1 x R_q x R_r x 1 x 1
    
    query_std = torch.sqrt((translated_rotated_query - query_mean).pow(2).sum(dim=(3,-2,-1), keepdim=True)) + 1e-7  # N x 1 x T x R_q x 1 x 1
    ref_std = torch.sqrt((sliced_ref - ref_mean).pow(2).sum(dim=(3,-2,-1), keepdim=True)) + 1e-7  # 1 x M x 1 x R_q x R_r x 1 x 1
    

    corr = (((translated_rotated_query - query_mean) * (sliced_ref - ref_mean)).sum(dim=(3,-2,-1), keepdim=True) / query_std / ref_std).mean(dim=(3,-2,-1)) # N x M x T x R_r
    br_corr, bestrots = corr.max(dim=-1) # N x M x T
    bt_corr, besttrans = torch.max(br_corr, dim=-1) # N x M
    bestrot = bestrots.gather(-1, besttrans.unsqueeze(-1)) #N x M x 1

    bestcorr, bestref = torch.max(bt_corr, dim=-1) # N

    # Combine best indices into a single Nx3 tensor
    best_indices = torch.stack([bestref, besttrans[torch.arange(N), bestref], bestrot[torch.arange(N), bestref][..., 0]], dim=1)


    return bestcorr, best_indices, corr, translated_rotated_query.view(N, T, R_q, D, D)[torch.arange(N), besttrans[torch.arange(N), bestref]], sliced_ref.view(M, R_q, R_r, D, D)[bestref, :, bestrot[torch.arange(N), bestref]]



def optimize_rot_trans_chunked(ref_maps, query_maps, query_rotation_matrices, ref_rotation_offsets, translation_vectors, chunk_size=100, hartley_corr=True):
    """
    Memory-efficient optimization of rotation and translation for query and reference maps.

    Args:
        ref_maps: Tensor of shape M x D x D x D or list of paths to pickle files containing reference maps
        query_maps: Tensor of shape N x D x D x D, query maps 
        query_rotation_matrices: Tensor of shape R_q x 3 x 3, rotation matrices for query maps
        ref_rotation_offsets: Tensor of shape R_r x 3 x 3, rotation offsets for reference maps
        translation_vectors: Tensor of shape T x 3, translation vectors
        chunk_size: Number of reference maps to process at once
        hartley_corr: bool, optional (default=True) compute correlations in Hartley space using symmetrized Hartley transforms.

    Returns:
        best_corr: shape N, best correlation for each query image
        best_indices: shape N x 3, indices of best (reference, translation, rotation) for each query
        corr: Tensor of shape N x M x T x R_r, correlation values
    """
    N = query_maps.shape[0]
    D = query_maps.shape[1]
    device = query_maps.device
    
    # Handle both tensor and list of paths
    if isinstance(ref_maps, list):
        chunk_files = ref_maps
    else:
        M = ref_maps.shape[0]
        chunk_files = [None]
    
    # Initialize arrays to store best results
    best_corr = torch.full((N,), float('-inf'))
    best_indices = torch.zeros((N, 3), dtype=torch.long)
    corr_all = []
    
    if hartley_corr:
        query_maps = symmetrize_ht3(torch.stack([htn_center(q) for q in query_maps]))

    # Pre-allocate a chunk of memory
    chunk_refs = torch.empty(chunk_size, D+1, D+1, D+1, device=device) if hartley_corr else torch.empty(chunk_size, D, D, D, device=device)
    
    # Process reference maps in chunks
    global_offset = 0
    for cf in chunk_files:
        if cf is not None:
            data = pickle.load(open(cf, 'rb'))
            ref_maps = data['maps']
            M = ref_maps.shape[0]
            
        for chunk_start in range(0, M, chunk_size):
            chunk_end = min(chunk_start + chunk_size, M)
            chunk_size_actual = chunk_end - chunk_start
            
            # Load chunk of reference maps to gpu
            if hartley_corr:
                chunk_refs[:chunk_size_actual].copy_(symmetrize_ht3(torch.stack([htn_center(r) for r in ref_maps[chunk_start:chunk_end]])))
            else:
                chunk_refs[:chunk_size_actual].copy_(ref_maps[chunk_start:chunk_end])


            # Process chunk
            chunk_best_vals, chunk_best_indices, corr, _, _ = optimize_rot_trans(
                                                            chunk_refs[:chunk_size_actual], 
                                                            query_maps,
                                                            query_rotation_matrices,
                                                            ref_rotation_offsets,
                                                            translation_vectors)
            chunk_best_vals = chunk_best_vals.cpu()
            chunk_best_indices = chunk_best_indices.cpu()
            corr = corr.cpu()

            # Adjust reference indices to account for chunking
            chunk_best_indices[:,0] += chunk_start + global_offset

            # Update best results where this chunk had better correlations
            better_mask = chunk_best_vals > best_corr
            best_corr[better_mask] = chunk_best_vals[better_mask]
            best_indices[better_mask] = chunk_best_indices[better_mask]

            corr_all.append(corr)

        global_offset += M

            
    # Combine results from all chunks
    return best_corr, best_indices, torch.cat(corr_all, dim=1)

import torch
from cryodrgn.lattice import Lattice
from cryodrgn.pose_search import rot_2d, interpolate
from cryodrgn import fft


def translate_images(images, trans, lat=None, mask=None, input_hartley=True, output_hartley=True):
    """
    Translate images in Hartley/real space

    Args:
        images: Tensor of shape N x D x D, images in real or Hartley space
        trans: Tensor of translations to apply, T x 2 or N x T x 2
        lat: Optional Lattice object. If None, creates new Lattice with D+1 size
        mask: Optional mask for lattice coordinates. If None, uses circular mask
        input_hartley: Bool indicating if input images are in Hartley space
        output_hartley: Bool indicating if output should be in Hartley space

    Returns:
        Tensor of translated images with shape N x T x D x D, where T is number of translations.
        Output is in Hartley or real space based on output_hartley parameter.
    """

    if lat is None:
        lat = Lattice(images.shape[1]+1) #, extent=images.shape[1]//2)

    if not input_hartley:
        ht = fft.ht2_center(images)
        ht = fft.symmetrize_ht(ht)
    else:
        ht = images

    if len(trans.shape) == 2:
        trans = trans.unsqueeze(0)

    if mask is None:
        mask = lat.get_circular_mask((images.shape[1]) // 2)

    translated_flat = lat.translate_ht(ht.reshape(images.shape[0], -1)[:, mask], trans, mask)

    # outputs of translation in hartley space. Shape N x T x D x D
    trans_images = torch.zeros((images.shape[0], translated_flat.shape[1], ht.shape[1]**2), device=images.device)
    trans_images[..., mask] = translated_flat

    if not output_hartley:
        trans_images = fft.iht2_center(trans_images.view((trans_images.shape[:-1] + ht.shape[1:]))[...,:-1,:-1])

    return trans_images

def rotate_images(images, rot, lat=None, mask=None, input_hartley=True, output_hartley=True, fast_rotate=False):
    """
    images - shape N x D x D, real space image
    rot - shape R, radians
    lat - optional Lattice object
    mask - optional mask for lattice
    input_hartley - bool, whether input is in hartley space
    output_hartley - bool, whether output should be in hartley space
    """
    if lat is None:
        lat = Lattice(images.shape[1]+1)

    if mask is None:
        mask = lat.get_circular_mask((images.shape[1]) // 2)

    if not input_hartley:
        ht = fft.ht2_center(images)
        ht = fft.symmetrize_ht(ht)
    else:
        ht = images

    if fast_rotate:
        # Rotate in hartley space
        rot_matrices = torch.stack([rot_2d(a, 2, ht.device) for a in rot], dim=0)
        lattice_coords = lat.coords[mask][:, :2] 
        rot_coords = lattice_coords @ rot_matrices

        ht_flat = ht.view(ht.shape[0], -1)[:, mask]

        full_images_masked = torch.zeros_like(ht)
        full_images_masked.view(-1, ht.shape[-1]**2)[:, mask] = ht_flat

        rot_images = torch.zeros((ht.shape[0], len(rot), ht.shape[-1], ht.shape[-1]), device=ht.device)

        for angle_idx, interp_coords in enumerate(rot_coords):
            interpolated = interpolate(full_images_masked, interp_coords)
            interpolated *= ht_flat.std(-1, keepdim=True) / interpolated.std(-1, keepdim=True)
            rot_images[:, angle_idx][:,mask] = interpolated
    else:
        rot_images = lat.rotate(ht, rot) # M x R x D x D

    if not output_hartley:
        rot_images = fft.iht2_center(rot_images[...,:-1,:-1])

    return rot_images

def optimize_theta_trans(ref_images, query_images, trans, rot, fast_rotate=False):
    """
    query_image - shape N x D x D, real space image
    ref_images - shape M x D x D, real space image
    trans - shape N x T x 2 or T x 2, cartesian
    rot - shape R, radians
    fast_rotate - do interpolation-based rotation, bool
    """

    if len(trans.shape) == 2:
        trans = trans.unsqueeze(0)

    lat = Lattice(ref_images.shape[1]+1)
    mask = lat.get_circular_mask((ref_images.shape[1]) // 2)

    # make many translations of the references in hartley space
    ref_ht = fft.ht2_center(ref_images)
    ref_ht = fft.symmetrize_ht(ref_images)

    # outputs of translation in hartley space. Shape N x T x D x D
    ref_trans_images = translate_images(ref_ht, trans, lat, mask)

    # rotate the query in hartley space 
    query_ht = fft.ht2_center(query_images)
    query_ht = fft.symmetrize_ht(query_images)

    # Use rotate_images function instead of duplicating rotation logic
    query_rot_images = rotate_images(query_ht, rot, lat=lat, mask=mask, 
                                   input_hartley=True, output_hartley=True,
                                   fast_rotate=fast_rotate)

    query_expanded = query_rot_images.unsqueeze(0).unsqueeze(2)
    ref_expanded = ref_trans_images.unsqueeze(1).unsqueeze(3)

    # Compute normalized cross correlation in hartley space
    pairwise_corr = (query_expanded * ref_expanded).sum(dim=(-1,-2)) / (
        torch.std(query_expanded, dim=(-1,-2)) * torch.std(ref_expanded, dim=(-1,-2)))

    return pairwise_corr
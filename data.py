import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from cryodrgn import ctf
from cryodrgn.source import ImageSource
from cryodrgn.lattice import Lattice
from proj_search import rotate_images, translate_images
import pickle

# corrupt a batch of particle images with simulated noise
def corrupt_with_ctf(batch_ptcls, batch_ctf_params, snr1, snr2, freqs, b_factor=None):

    stds = batch_ptcls.std(dim=(-1,-2))

    batch_ptcls += torch.randn(batch_ptcls.shape) * (stds / torch.sqrt(snr1)).view([-1,1,1])


    batch_ptcls = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(batch_ptcls, dim=(-1, -2))), dim=(-1, -2))
    batch_ptcls = batch_ptcls.real - batch_ptcls.imag  # centered hartley transform
    ctf_weights = ctf.compute_ctf(freqs, *torch.split(batch_ctf_params, 1, dim=1), bfactor=b_factor).view(-1, batch_ptcls.shape[-1], batch_ptcls.shape[-1])
    batch_ptcls *= ctf_weights
    batch_ptcls = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(batch_ptcls, dim=(-1, -2))), dim=(-1, -2))
    batch_ptcls = batch_ptcls.real - batch_ptcls.imag  

    batch_ptcls += torch.randn(batch_ptcls.shape)  * (stds / torch.sqrt(snr2)).view([-1,1,1])
    return batch_ptcls


class ContrastiveProjectionDataset(Dataset):
    def __init__(self, image_files, phis, thetas, object_ids, pos_angle_threshold=30, pclean=0.3, snr1=[1.5], 
                 dfu=[10000], Apix=5.0, ang=0.0, kv=300, cs=2.7, wgh=0.1, ps=0.0, proj_per_obj=192, img_size=128):
        """
        Dataset for contrastive learning of image projections.
        
        Args:
            image_files: List of file paths for each projection image
            phis: Tensor of shape N containing phi angles in degrees
            thetas: Tensor of shape N containing theta angles in degrees  
            object_ids: Tensor of shape N containing integer IDs for each 3D object
            pos_angle_threshold: Maximum angular difference in degrees for positive pairs
            snr1: List of signal-to-noise ratios for structural noise. snr2 is snr1 / 3
            dfu: List of defocus values in Angstroms for u-axis. dfv is dfu + 500
            Apix: Pixel size in Angstroms
            ang: Astigmatism angle in degrees
            kv: Microscope voltage in kV
            cs: Spherical aberration in mm
            wgh: Amplitude contrast ratio
            ps: Phase shift in radians
        """
        self.image_files = image_files
        self.phis = phis 
        self.thetas = thetas
        self.object_ids = object_ids
        self.pos_threshold = pos_angle_threshold
        self.snr1 = snr1
        self.dfu = dfu
        self.Apix = Apix
        self.ang = ang
        self.kv = kv
        self.cs = cs
        self.wgh = wgh
        self.ps = ps
        self.pclean = pclean
        self.proj_per_obj = proj_per_obj
        self.img_size = img_size

        # Create a Lattice object for transformations
        self.lat = Lattice(img_size + 1)
        self.mask = self.lat.get_circular_mask(img_size // 2)

        # prepare frequency lattice
        freqs = torch.arange(-img_size//2, img_size//2) / (self.Apix * img_size)
        x0, x1 = torch.meshgrid(freqs, freqs)
        self.freqs = torch.stack([x0.ravel(), x1.ravel()], dim=1)

    def __len__(self):
        return len(self.image_files) * self.proj_per_obj

    def __getitem__(self, idx):
        """
        Returns:
            anchor_img: The anchor image
            pos_img: A positive pair image (same object, similar angle)
            neg_img: A negative pair image (different object)
            pos_dist: Angular distance between anchor and positive pair
        """
        # Load images from files
        obj_idx = idx // self.proj_per_obj
        proj_id = idx % self.proj_per_obj

        image_stack = ImageSource.from_file(self.image_files[obj_idx] + '.mrcs').images() 
        anchor_img = image_stack[proj_id]
        anchor_obj = self.object_ids[obj_idx]
        
        phis, thetas = pickle.load(open(self.image_files[obj_idx] + '_pose.pkl','rb')) 


        # Get positive pair from same image stack with similar viewing angle
        # Calculate angular distances between anchor and all other projections using spherical coordinates
        # cos(angle) = sin(theta1)sin(theta2)cos(phi1-phi2) + cos(theta1)cos(theta2)
        theta1, phi1 = thetas[proj_id], phis[proj_id]
        angle_dists = torch.arccos(
            torch.sin(theta1) * torch.sin(thetas) * torch.cos(phi1 - phis) + 
            torch.cos(theta1) * torch.cos(thetas)
        )
        
        # Create mask of valid positive pairs (within threshold and not same projection)
        valid_mask = (angle_dists <= self.pos_threshold * torch.pi/180)
        
        # Sample from valid indices
        valid_indices = torch.where(valid_mask)[0]
        pos_idx = valid_indices[torch.randint(len(valid_indices), (1,))].item()
        pos_img = image_stack[pos_idx]
        
        # Get negative pair from different object
        neg_obj_idx = torch.randint(len(self.image_files), (1,)).item()
        while neg_obj_idx == obj_idx:  # Ensure different object
            neg_obj_idx = torch.randint(len(self.image_files), (1,)).item()
            
        neg_stack = ImageSource.from_file(self.image_files[neg_obj_idx] + '.mrcs').images()
        neg_proj_idx = torch.randint(neg_stack.shape[0], (1,)).item()
        neg_img = neg_stack[neg_proj_idx]
        neg_obj = self.object_ids[neg_obj_idx]

        # Sample CTF parameters for anchor, positive and negative images
        anchor_ctf = torch.zeros(9)
        pos_ctf = torch.zeros(9)
        neg_ctf = torch.zeros(9)

        # Set D and pixel size (first two params)
        anchor_ctf[0] = self.img_size
        anchor_ctf[1] = self.Apix
        pos_ctf[0] = self.img_size
        pos_ctf[1] = self.Apix
        neg_ctf[0] = self.img_size
        neg_ctf[1] = self.Apix


        # Randomly sample remaining CTF params for each image
        for i, param_list in enumerate([self.snr1, 0.0, self.dfu, 0.0, self.ang, self.kv,
                                      self.cs, self.wgh, self.ps]):
            if isinstance(param_list, list):
                anchor_ctf[i] = param_list[torch.randint(len(param_list), (1,))]
                pos_ctf[i] = param_list[torch.randint(len(param_list), (1,))]
                neg_ctf[i] = param_list[torch.randint(len(param_list), (1,))]
            else:
                anchor_ctf[i] = param_list
                pos_ctf[i] = param_list
                neg_ctf[i] = param_list

        for ctf in (anchor_ctf, pos_ctf, neg_ctf):
            ctf[1] = ctf[0] / 3
            ctf[3] = ctf[2] + 500

        # Stack images and CTF params
        images = torch.stack([anchor_img, pos_img, neg_img], dim=0)
        ctf_params = torch.stack([anchor_ctf, pos_ctf, neg_ctf], dim=0)

        return images, ctf_params, torch.tensor([anchor_obj, anchor_obj, neg_obj])

    @staticmethod
    def collate_fn(batch, lat, mask, freqs, corrupt=False, zoom=False):
        """
        Custom collate function to corrupt batches of triplet images with CTF and noise
        Args:
            batch: List of (images, ctf_params, pos_dist) tuples
        Returns:
            Corrupted (images, ctf_params, pos_dist) tuples
        """
        # Stack all images and CTF params from batch
        images = torch.stack([x[0] for x in batch])  # Shape: (batch, 3, D, D)
        ctf_params = torch.stack([x[1] for x in batch])  # Shape: (batch, 3, 9) 
        obj_ids = torch.stack([x[2] for x in batch])  # Shape: (batch,)

        B, N, D, _ = images.shape
        
        # Reshape to (batch*3, D, D) for corruption
        images = images.view(-1, D, D)
        ctf_params = ctf_params.view(-1, 9)

        # Corrupt part of batch
        if corrupt:
            corrupted = corrupt_with_ctf(images, ctf_params[:,2:], ctf_params[:,0], ctf_params[:,1], freqs)
        else:
            corrupted = images


        # Apply random rotations and translations after corruption
        corrupted = corrupted.view(-1, D, D)

        # Randomly generate rotations and translations
        rotations = torch.rand(B * N) * 2 * torch.pi  # Random rotations in radians
        translations = torch.randint(-7, 8, (B * N, 2)).to(torch.float)  # Random translations within a range

        # Apply random rotations and translations
        corrupted = rotate_images(corrupted, rotations, lat=lat, mask=mask, input_hartley=False, output_hartley=False).diagonal().permute([2,0,1])
        
        corrupted = translate_images(corrupted, translations, lat=lat, mask=mask, input_hartley=False, output_hartley=False).diagonal().permute([2,0,1])
        
        # Reshape back to (batch, 3, D, D)
        corrupted = corrupted.view(B, N, D, D)

        ctf_params = ctf_params.view(B, N, 9)

        corrupted = (corrupted - corrupted.mean(dim=(-1,-2), keepdim=True)) / corrupted.std(dim=(-1,-2), keepdim=True)
    

        return corrupted, ctf_params, obj_ids

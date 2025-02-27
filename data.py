import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from cryodrgn import ctf
from cryodrgn.lattice import Lattice
from proj_search import rotate_images, translate_images


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
    def __init__(self, images, phis, thetas, object_ids, pos_angle_threshold=30, pclean=0.3, snr1=[1.5], 
                 dfu=[10000], Apix=5.0, ang=0.0, kv=300, cs=2.7, wgh=0.1, ps=0.0):
        """
        Dataset for contrastive learning of image projections.
        
        Args:
            images: Tensor of shape N x H x W containing projection images
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
        self.images = images
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

        # Pre-compute angular distances between all pairs
        self.angular_dists = self._compute_angular_distances()
        
        # Get indices grouped by object ID for efficient negative sampling
        self.obj_to_idx = {}
        for i, obj_id in enumerate(object_ids):
            if obj_id.item() not in self.obj_to_idx:
                self.obj_to_idx[obj_id.item()] = []
            self.obj_to_idx[obj_id.item()].append(i)

        # Create a Lattice object for transformations
        self.lat = Lattice(images.shape[-1] + 1)
        self.mask = self.lat.get_circular_mask((images.shape[-1]) // 2)

        # prepare frequency lattice
        freqs = torch.arange(-images.shape[-1]//2, images.shape[-1]//2) / (self.Apix * images.shape[-1])
        x0, x1 = torch.meshgrid(freqs, freqs)
        self.freqs = torch.stack([x0.ravel(), x1.ravel()], dim=1)


    def _compute_angular_distances(self):
        """Compute pairwise angular distances between all poses"""
        N = len(self.phis)
        dists = torch.zeros((N, N))
        
        # Compute 3D unit vectors for each orientation
        x = torch.cos(self.phis) * torch.sin(self.thetas)
        y = torch.sin(self.phis) * torch.sin(self.thetas)
        z = torch.cos(self.thetas)
        vectors = torch.stack([x, y, z], dim=1)
        
        # Compute angular distances using dot product
        dists = torch.rad2deg(torch.arccos(torch.clamp(
            torch.matmul(vectors, vectors.T), -1.0, 1.0)))
            
        return dists

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            anchor_img: The anchor image
            pos_img: A positive pair image (same object, similar angle)
            neg_img: A negative pair image (different object)
            pos_dist: Angular distance between anchor and positive pair
        """
        anchor_img = self.images[idx]
        anchor_obj = self.object_ids[idx]
        
        # Find valid positive pairs (same object, within angle threshold)
        pos_mask = (self.object_ids == anchor_obj) & (self.angular_dists[idx] <= self.pos_threshold) & (torch.arange(len(self.images)) != idx)
        neg_mask = torch.logical_not(pos_mask)

        # Randomly select positive pair
        pos_idx = torch.randint(pos_mask.sum(), (1,)).item()
        pos_img = self.images[pos_mask][pos_idx]
        pos_dist = self.angular_dists[idx, pos_mask][pos_idx]
        
        # Randomly select negative pair from different object
        neg_img = self.images[neg_mask][torch.randint(neg_mask.sum(), (1,)).item()]

        # Sample CTF parameters for anchor, positive and negative images
        anchor_ctf = torch.zeros(9)
        pos_ctf = torch.zeros(9)
        neg_ctf = torch.zeros(9)

        # Set D and pixel size (first two params)
        anchor_ctf[0] = self.images.shape[-1]
        anchor_ctf[1] = self.Apix
        pos_ctf[0] = self.images.shape[-1]
        pos_ctf[1] = self.Apix
        neg_ctf[0] = self.images.shape[-1]
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
            ctf[4] = ctf[3] + 500

        # Stack images and CTF params
        images = torch.stack([anchor_img, pos_img, neg_img], dim=0)
        ctf_params = torch.stack([anchor_ctf, pos_ctf, neg_ctf], dim=0)

        return images, ctf_params, pos_dist

    @staticmethod
    def collate_fn(batch, lat, mask, freqs, pclean=0.4):
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
        pos_dists = torch.stack([x[2] for x in batch])  # Shape: (batch,)

        B, N, D, _ = images.shape
        
        # Reshape to (batch*3, D, D) for corruption
        images = images.view(-1, D, D)
        ctf_params = ctf_params.view(-1, 9)

        # Corrupt the full batch
        if torch.rand(1).item() > pclean:
            corrupted = corrupt_with_ctf(images, ctf_params[:,2:], ctf_params[:,0], ctf_params[:,1], freqs)

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

        #Apply random zoom and crop
        D = corrupted.shape[-1]
        max_zoom = 1.15  # 15% zoom
        
        #Generate random zoom factors for each image, 50% chance of no zoom
        zoom_mask = (torch.rand(B, N) > 0.5).float()  # 1 for zoom, 0 for no zoom
        zoom_factors = 1 + zoom_mask * (torch.rand(B, N) * (max_zoom - 1))  # Shape: (batch, 3)

        #Initialize tensor for zoomed images
        zoomed = torch.zeros_like(corrupted)
        
        for b in range(B):
           for n in range(N):
               #Calculate new size after zoom
               zoom = zoom_factors[b,n]
               new_size = int(D * zoom)
                
               #Resize image
               zoomed_img = F.interpolate(corrupted[b,n].unsqueeze(0).unsqueeze(0), 
                                        size=(new_size, new_size),
                                        mode='bilinear',
                                        align_corners=False)
                
               #Calculate crop boundaries to get back to original size
               start = (new_size - D) // 2
               end = start + D
                
               #Crop and store
               zoomed[b,n] = zoomed_img[0,0,start:end,start:end]
        
        #corrupted = zoomed
        ctf_params = ctf_params.view(B, N, 9)

        return corrupted, ctf_params, pos_dists

import logging
import torch
import clip
import open_clip

logger = logging.getLogger(__name__)

# ==============================================================================
# SINGLETON BACKBONE REGISTRY
# Acts as a centralized manager for massive vision models. Prevents redundant 
# VRAM allocation by ensuring each CLIP architecture is only loaded once in FP16.
# ==============================================================================
class BackboneRegistry:
    _vit_l_14 = None
    _vit_l_14_preprocess = None
    
    _vit_h_14 = None
    _vit_h_14_preprocess = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_vit_l_14(cls):
        """
        Returns the shared OpenAI ViT-L/14 backbone and preprocessor.
        Used by Simulacra, LAION_Aesthetic, etc.
        """
        if cls._vit_l_14 is None:
            logger.info("[BACKBONE REGISTRY] Initializing shared ViT-L/14 in FP16 (~1.5GB VRAM)...")
            model, preprocess = clip.load("ViT-L/14", device=cls.device)
            # Slash VRAM footprint by 50% and set to evaluation mode
            cls._vit_l_14 = model.eval()
            cls._vit_l_14_preprocess = preprocess
            
        return cls._vit_l_14, cls._vit_l_14_preprocess

    # @classmethod
    # def get_vit_h_14(cls):
    #     """
    #     Returns the shared OpenCLIP ViT-H-14 (LAION-2B) backbone and preprocessor.
    #     Used by custom MPS implementations, PickScore, etc.
    #     """
    #     if cls._vit_h_14 is None:
    #         logger.info("[BACKBONE REGISTRY] Initializing shared ViT-H-14 in FP16 (~2.5GB VRAM)...")
    #         model, _, preprocess = open_clip.create_model_and_transforms(
    #             "ViT-H-14", 
    #             pretrained="laion2b_s32b_b79k", 
    #             device=cls.device
    #         )
    #         # Slash VRAM footprint by 50% and set to evaluation mode
    #         cls._vit_h_14 = model.eval()
    #         cls._vit_h_14_preprocess = preprocess
            
    #     return cls._vit_h_14, cls._vit_h_14_preprocess

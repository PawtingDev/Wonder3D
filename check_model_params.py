from diffusers import DDPMScheduler, AutoencoderKL, EMAModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

# base model
pretrained_model_name_or_path = 'flamehaze1115/wonder3d-v1.0'
# fine-tuning model
pretrained_unet_path = 'outputs/wonder3D-joint-0327/unet-15000'
# pretrained_unet_path = 'outputs/wonder3D-joint/unet-150'
# pretrained_unet_path = 'outputs/wonder3D-joint/checkpoint'
# pretrained_unet_path = 'flamehaze1115/wonder3d-v1.0'

trainable_modules = ('joint_mid')
revision = 'main'
unet_from_pretrained_kwargs = {
    'camera_embedding_type': 'e_de_da_sincos',
    'projection_class_embeddings_input_dim': 10,  # modify
    'num_views': 6,
    'sample_size': 32,
    'zero_init_conv_in': False,
    'zero_init_camera_projection': False,
    'cd_attention_last': False,
    'cd_attention_mid': True,
    'multiview_attention': True,
    'sparse_mv_attention': False,
    'mvcd_attention': False
}

noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path,
                                                              subfolder="image_encoder", revision=revision)
feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path, subfolder="feature_extractor",
                                                       revision=revision)
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
print("load pre-trained unet from ", pretrained_unet_path)
unet = UNetMV2DConditionModel.from_pretrained(pretrained_unet_path, subfolder="unet", revision=revision,
                                              **unet_from_pretrained_kwargs)

ema_unet = EMAModel(unet.parameters(), model_cls=UNetMV2DConditionModel, model_config=unet.config)

for name, module in unet.named_modules():
    if name.endswith(trainable_modules):
        # print(name)
        """
        down_blocks.0.attentions.0.transformer_blocks.0.attn_joint_mid
        down_blocks.0.attentions.0.transformer_blocks.0.norm_joint_mid
        down_blocks.0.attentions.1.transformer_blocks.0.attn_joint_mid
        down_blocks.0.attentions.1.transformer_blocks.0.norm_joint_mid
        down_blocks.1.attentions.0.transformer_blocks.0.attn_joint_mid
        down_blocks.1.attentions.0.transformer_blocks.0.norm_joint_mid
        down_blocks.1.attentions.1.transformer_blocks.0.attn_joint_mid
        down_blocks.1.attentions.1.transformer_blocks.0.norm_joint_mid
        down_blocks.2.attentions.0.transformer_blocks.0.attn_joint_mid
        down_blocks.2.attentions.0.transformer_blocks.0.norm_joint_mid
        down_blocks.2.attentions.1.transformer_blocks.0.attn_joint_mid
        down_blocks.2.attentions.1.transformer_blocks.0.norm_joint_mid
        up_blocks.1.attentions.0.transformer_blocks.0.attn_joint_mid
        up_blocks.1.attentions.0.transformer_blocks.0.norm_joint_mid
        up_blocks.1.attentions.1.transformer_blocks.0.attn_joint_mid
        up_blocks.1.attentions.1.transformer_blocks.0.norm_joint_mid
        up_blocks.1.attentions.2.transformer_blocks.0.attn_joint_mid
        up_blocks.1.attentions.2.transformer_blocks.0.norm_joint_mid
        up_blocks.2.attentions.0.transformer_blocks.0.attn_joint_mid
        up_blocks.2.attentions.0.transformer_blocks.0.norm_joint_mid
        up_blocks.2.attentions.1.transformer_blocks.0.attn_joint_mid
        up_blocks.2.attentions.1.transformer_blocks.0.norm_joint_mid
        up_blocks.2.attentions.2.transformer_blocks.0.attn_joint_mid
        up_blocks.2.attentions.2.transformer_blocks.0.norm_joint_mid
        up_blocks.3.attentions.0.transformer_blocks.0.attn_joint_mid
        up_blocks.3.attentions.0.transformer_blocks.0.norm_joint_mid
        up_blocks.3.attentions.1.transformer_blocks.0.attn_joint_mid
        up_blocks.3.attentions.1.transformer_blocks.0.norm_joint_mid
        up_blocks.3.attentions.2.transformer_blocks.0.attn_joint_mid
        up_blocks.3.attentions.2.transformer_blocks.0.norm_joint_mid
        mid_block.attentions.0.transformer_blocks.0.attn_joint_mid
        mid_block.attentions.0.transformer_blocks.0.norm_joint_mid
        """

        for params in module.parameters():
            print("trainable params: ", params, params.shape)
            exit()
            # params.requires_grad = True

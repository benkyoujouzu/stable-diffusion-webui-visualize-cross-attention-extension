import os
import gradio as gr
from modules import scripts, script_callbacks
from PIL import Image
import numpy as np
import torch
import modules.shared as shared
from modules import devices
from torch import nn, einsum
from einops import rearrange
import math
from ldm.modules.attention import CrossAttention

    
hidden_layers = {}
hidden_layer_names = []
default_hidden_layer_name = "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2"
hidden_layer_select = None

def update_layer_names(model):
    hidden_layers = {}
    for n, m in model.named_modules():
        if(isinstance(m, CrossAttention)):
            hidden_layers[n] = m
    hidden_layer_names = list(filter(lambda s : "attn2" in s, hidden_layers.keys())) 
    if hidden_layer_select != None:
        hidden_layer_select.update(value=default_hidden_layer_name, choice=hidden_layer_names)

def get_attn(emb, ret):
    def hook(self, sin, sout):
        h = self.heads
        q = self.to_q(sin[0])
        context = emb
        k = self.to_k(context)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        ret["out"] = attn
    return hook

def generate_vxa(image, prompt, idx, time, layer_name, output_mode):
    if(not isinstance(image, np.ndarray)):
        return image
    output = image.copy()
    image = image.astype(np.float32) / 255.0
    image = np.moveaxis(image, 2, 0)
    image = torch.from_numpy(image).unsqueeze(0)

    model = shared.sd_model
    layer = hidden_layers[layer_name]
    cond_model = model.cond_stage_model
    with torch.no_grad(), devices.autocast():
        image = image.to(devices.device)
        latent = model.get_first_stage_encoding(model.encode_first_stage(image))
        try:
            t = torch.tensor([float(time)]).to(devices.device)
        except:
            return output
        emb = cond_model([prompt])

        attn_out = {}
        handle = layer.register_forward_hook(get_attn(emb, attn_out))
        try:
            model.apply_model(latent, t, emb)
        finally:
            handle.remove()

    if (idx == ""):
        img = attn_out["out"][:,:,1:].sum(-1).sum(0)
    else:
        try:
            idxs = list(map(int, filter(lambda x : x != '', idx.strip().split(','))))
            img = attn_out["out"][:,:,idxs].sum(-1).sum(0)
        except:
            return output

    scale = round(math.sqrt((image.shape[2] * image.shape[3]) / img.shape[0]))
    h = image.shape[2] // scale
    w = image.shape[3] // scale
    img = img.reshape(h, w) / img.max()
    img = img.to("cpu").numpy()
    output = output.astype(np.float64)
    if output_mode == "masked":
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i][j] *= img[i // scale][j // scale]
    elif output_mode == "grey":
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i][j] = [img[i // scale][j // scale] * 255.0] * 3
    output = output.astype(np.uint8)
    return output

def add_tab():     
    with gr.Blocks(analytics_enabled=False) as visualize_cross_attention:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(elem_id="vxa_input_image")
                vxa_prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Prompt to be visualized")
                vxa_token_indices = gr.Textbox(value="", label="Indices of tokens to be visualized", lines=2, placeholder="Example: 1, 3 means the sum of the first and the third tokens. 1 is suggected for a single token. Leave blank to visualize all tokens.")
                vxa_time_embedding = gr.Textbox(value="1.0", label="Time embedding")
                for n, m in shared.sd_model.named_modules():
                    if(isinstance(m, CrossAttention)):
                        hidden_layers[n] = m
                hidden_layer_names = list(filter(lambda s : "attn2" in s, hidden_layers.keys())) 
                hidden_layer_select = gr.Dropdown(value=default_hidden_layer_name, label="Cross-attention layer", choices=hidden_layer_names)
                vxa_output_mode = gr.Dropdown(value="masked", label="Output mode", choices=["masked", "grey"])
                vxa_generate = gr.Button(value="Visualize Cross-Attention", elem_id="vxa_gen_btn")
            with gr.Column():
                vxa_output = gr.Image(elem_id = "vxa_output", interactive=False)
    
        vxa_generate.click(
            fn=generate_vxa,
            inputs=[input_image, vxa_prompt, vxa_token_indices, vxa_time_embedding, hidden_layer_select, vxa_output_mode],
            outputs=[vxa_output],
        )

    return (visualize_cross_attention, "VXA", "visualize_cross_attention"),

script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_model_loaded(update_layer_names)


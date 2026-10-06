"""UNTRAINED, opt-in polyphase capacity/initialization control, NOT a trained VAE.

All raw points pass through the real 8x encoder, latent and decoder. No raw-input
skip or target/source-ID oracle. Structured weights pack/unpack40 scalar fields;
zero residual-output projections leave a near-identity route. Anchored LayerNorm
induces a small known nonlinearity. OCR/style/generative usefulness and stability
under subsequent optimization are NOT established. Never installed by default.
"""
import math
import torch

def initialize_identity_geometry(model,S=512.,std=2e-5):
 from model.blocks import Residual
 if not math.isfinite(S) or S<128 or not 0<std<=.01:raise ValueError("bounded feature scale and active posterior noise required")
 if model.encoder.conv_1.in_channels!=5 or model.conv_mu.out_channels<40 or model.transformer_decoder.fc.out_features!=123:raise ValueError("five input fields, at least40 latent channels, and20 mixtures required")
 if model.transformer_decoder.input_proj.out_features<=10 or model.transformer_decoder.input_proj.out_features%2:raise ValueError("even Transformer width with anchor channels required")
 fields=5
 for layer in [model.encoder.conv_1,model.encoder.conv_2,model.encoder.conv_3]:
  if layer.in_channels<fields or layer.out_channels<2*fields or layer.kernel_size!=(3,) or layer.stride!=(2,) or layer.padding!=(1,):raise ValueError('compatible stride2 polyphase encoder required')
  fields*=2
 if model.conv_mu.in_channels<fields or model.conv_logvar.weight.shape!=model.conv_mu.weight.shape:raise ValueError('compatible40-field latent heads required')
 for layer in [model.decoder.deconv_1,model.decoder.deconv_2,model.decoder.deconv_3]:
  if layer.in_channels<fields or layer.out_channels<fields//2 or layer.kernel_size!=(4,) or layer.stride!=(2,) or layer.padding!=(1,) or layer.output_padding!=(0,):raise ValueError('compatible stride2 polyphase decoder required')
  fields//=2
 with torch.no_grad():
  for container in (model.encoder,model.decoder):
   for module in container.modules():
    if isinstance(module,Residual):module.block[-1].weight.zero_()
  fields=5
  for layer in [model.encoder.conv_1,model.encoder.conv_2,model.encoder.conv_3]:
   layer.weight.zero_();layer.bias.zero_()
   for phase in range(2):
    for c in range(fields):layer.weight[phase*fields+c,c,1+phase]=1
   fields*=2
  model.conv_mu.weight.zero_();model.conv_mu.bias.zero_()
  for c in range(fields):model.conv_mu.weight[c,c,0]=1
  model.conv_logvar.weight.zero_();model.conv_logvar.bias.zero_();model.conv_logvar.bias[:fields]=2*math.log(std)
  for layer in [model.decoder.deconv_1,model.decoder.deconv_2,model.decoder.deconv_3]:
   fields//=2;layer.weight.zero_();layer.bias.zero_()
   for phase in range(2):
    for c in range(fields):layer.weight[phase*fields+c,c,1+phase]=1
  trans=model.transformer_decoder;trans.input_proj.weight.zero_();trans.input_proj.bias.zero_()
  width=trans.input_proj.out_features
  trans.input_proj.bias[10::2]=1;trans.input_proj.bias[11::2]=-1
  for c in range(5):trans.input_proj.weight[2*c,c]=1/S;trans.input_proj.weight[2*c+1,c]=-1/S
  gain=1.;v=(width-10)/width
  for layer in trans.transformer.layers:
   layer.self_attn.out_proj.weight.zero_();layer.self_attn.out_proj.bias.zero_();layer.linear2.weight.zero_();layer.linear2.bias.zero_()
   for norm in [layer.norm1,layer.norm2]:
    norm.weight.fill_(1);norm.bias.zero_();gain/=math.sqrt(gain*gain*v+norm.eps)
  trans.fc.weight.zero_();trans.fc.bias.zero_()
  def head(row,field,factor=1):
   trans.fc.weight[row,2*field]=factor*S/(2*gain);trans.fc.weight[row,2*field+1]=-factor*S/(2*gain)
  for k in range(20):head(23+k,0);head(43+k,1)
  for c in range(3):head(c,2+c,20)
 return dict(scale=S,std=std,packed_fields=40,learned=False,architecture_changed=False)

import torch
import torch.nn as nn
import fairseq
from conformer import ConformerBlock
from torch.nn.modules.transformer import _get_clones
from torch import Tensor

class MyConformer(nn.Module):
  def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1):
    super(MyConformer, self).__init__()
    self.dim_head=int(emb_size/heads)
    self.dim=emb_size
    self.heads=heads
    self.kernel_size=kernel_size
    self.n_encoders=n_encoders
    self.encoder_blocks=_get_clones( ConformerBlock( dim = emb_size, dim_head=self.dim_head, heads= heads, 
    ff_mult = ffmult, conv_expansion_factor = exp_fac, conv_kernel_size = kernel_size),
    n_encoders)
    self.class_token = nn.Parameter(torch.rand(1, emb_size))
    self.fc5 = nn.Linear(emb_size, 2)

  def forward(self, x, device): # x shape [bs, tiempo, frecuencia]
    x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])#[bs,1+tiempo,emb_size]
    for layer in self.encoder_blocks:
            x = layer(x) #[bs,1+tiempo,emb_size]
    embedding=x[:,0,:] #[bs, emb_size]
    out=self.fc5(embedding) #[bs,2]
    return out, embedding

class SSLModel(nn.Module): #W2V
    def __init__(self,device):
        super(SSLModel, self).__init__()
        cp_path = 'xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()      

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
                
        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V + Conformer')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer=MyConformer(emb_size=args.emb_size, n_encoders=args.num_encoders,
        heads=args.heads, kernel_size=args.kernel_size)
    def forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out, _ =self.conformer(x,self.device)
        return out

class Model2(nn.Module): #Variable len
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V + Conformer: Variable Length')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer=MyConformer(emb_size=args.emb_size, n_encoders=args.num_encoders,
        heads=args.heads, kernel_size=args.kernel_size)
    def forward(self, x): # x is a list of np arrays
        nUtterances = len(x)
        output = torch.zeros(nUtterances, 2).to(self.device)
        for n, feat in enumerate(x):
            input_x = torch.from_numpy(feat[:, :]).float().to(self.device)
            x_ssl_feat = self.ssl_model.extract_feat(input_x.squeeze(-1))
            f=self.LL(x_ssl_feat) 
            f = f.unsqueeze(dim=1)
            f = self.first_bn(f)
            f = self.selu(f)
            f = f.squeeze(dim=1)
            out, _ =self.conformer(f,self.device)
            output[n, :] = out
        return output

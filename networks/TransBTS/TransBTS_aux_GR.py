import torch
import torch.nn as nn
from .Transformer import TransformerModel
from .PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
from .Unet_skipconnection import Unet
# from networks.TransBTS.TransBTS_aux_GR import TransBTS
from networks.TransBTS.GradientReversal import *

class TransformerBTS(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(TransformerBTS, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim #128
        self.num_heads = num_heads
        self.patch_dim = patch_dim  #8
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 1) #256//16=16 --> 16*16=256
        self.seq_length = self.num_patches      # 256
        self.flatten_dim = 128 * num_channels   # 128

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:

            self.conv_x = nn.Conv2d(
                32,
                self.embedding_dim,
                kernel_size=3,
                stride=2,
                padding=1
            )

        self.Unet = Unet(in_channels=1, base_channels=4)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        #print("hidden dim in init", hidden_dim)  #256


    def encode(self, x):
        if self.conv_patch_representation:
            # combine embedding with conv patch distribution
            x1_1, x2_1, x3_1, x = self.Unet(x)
            #print("x after Unet:", x.shape) # [8, 32, 32, 32]
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv_x(x)
            #print("x after conv:", x.shape) # [8, 128, 16, 16]
            x = x.permute(0, 2, 3, 1).contiguous()
            #print("x after permute:", x.shape) # [8, 16, 16, 128]
            x = x.view(x.size(0), -1, self.embedding_dim)
            #print("conv respresentation is true:", x.shape) # [8, 256, 128]

        else:
            x1_1, x2_1, x3_1, x = self.Unet(x) 
            print("x shape after Unet in conv false representation ", x.shape) # [8, 32, 32, 32]
            x = self.bn(x)
            x = self.relu(x)
            x = (
                x.unfold(2, 2, 2)
                .unfold(3, 2, 2)
                .unfold(4, 2, 2)
                .contiguous()
            )
            print("x shape after unfold:", x.shape) # [8, 32, 16, 16, 1, 2, 2]
            x = x.view(x.size(0), x.size(1), -1, 8)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            print("x shape view:", x.shape) #[8, 256, 128]
            x = self.linear_encoding(x)
            print("conv representation is false:", x.shape) #[8, 256, 128]

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)
        #print("x shape in encoder", x.shape) #[8, 1024, 128]

        return x1_1, x2_1, x3_1, x, intmd_x

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x, auxillary_output_layers=[1, 2, 3, 4]):

        x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs = self.encode(x)

        decoder_output = self.decode(
            x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = 'Z' + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]

            return decoder_output

        return decoder_output

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        #print("x within reshape output:", x.shape) # [8, 256, 128]
        #print(x.size(0))
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            #int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        #print("within reshpe shape:", x.shape)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class BTS(TransformerBTS):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        #num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
        # gr_flag=False,
    ):
        super(BTS, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
            # gr_flag = gr_flag,
        )

        #self.num_classes = num_classes

        self.Softmax = nn.Softmax(dim=1)
        self.reversal_flag = False

        self.fc = nn.Sequential(
            nn.Linear(in_features=embedding_dim*hidden_dim, out_features=hidden_dim, bias=False),
            #nn.BatchNorm1d(num_features=hidden_dim, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=3, bias=False),
        )

        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim//4, out_channels=self.embedding_dim//8)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim//8)
        

        #self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim//8, out_channels=self.embedding_dim//16)
        self.convTrans3 = nn.ConvTranspose2d(in_channels=self.embedding_dim//8, out_channels=self.embedding_dim//16, kernel_size=4, stride=2, padding=1)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim//16)

        #self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim//16, out_channels=self.embedding_dim//32)
        self.convTrans2 = nn.ConvTranspose2d(in_channels=self.embedding_dim//16, out_channels=self.embedding_dim//32, kernel_size=4, stride=2, padding=1)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim//32)

        self.endconv = nn.Conv2d(self.embedding_dim // 32, 1, kernel_size=1)

    def set_reverse(self, mode):
        self.reversal_flag = mode

    def decode(self, x1_1, x2_1, x3_1, x, intmd_x, intmd_layers=[1, 2, 3, 4]):

        assert intmd_layers is not None, "pass the intermediate layers for MLA"
        encoder_outputs = {}
        all_keys = []
        for i in intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()

        # print(len(intmd_x)) -->8
        #print(x.shape) #[8, 256, 128]
        x = x.view(x.size(0), -1)
        #print(x.shape)
        z_out = self.fc(x)
        #print("z_out before sigmoid:", z_out.shape)     # [8, 256]
        #z_out2 = self.fc2(z_out) 
        #print("z_out2 before sigmoid:", z_out2.shape)   # [8, 3]

        x8 = encoder_outputs[all_keys[0]]
        #print("x8 before reshape:", x8.shape) # [8, 256, 128]
        x8 = self._reshape_output(x8)
        #print("x8 afrter reshape:", x8.shape)   # [8, 128, 16, 16]
        x8 = self.Enblock8_1(x8)    # [8, 32, 16, 16]
        #print("x8 after enblock8_1:", x8.shape)
        x8 = self.Enblock8_2(x8)    # [8, 32, 16, 16] add input from previous layer 
        #print("x8 after enblock8_2:", x8.shape)

        # print("x3_1 shape:", x3_1.shape) # [8, 16, 64, 64]
        y4 = self.DeUp4(x8, x3_1)    # (8, )
        #print("y4 after deup4 shape:", y4.shape)    # [8, 16, 64, 64]
        y4 = self.DeBlock4(y4)      # (8, )
        #print("y4 after deblock4 shape:", y4.shape)     # [8, 16, 64, 64]

        #y3 = self.DeUp3(y4, x2_1)  # (
        y3 = self.convTrans3(y4)
        #print("y3 shape:", y3.shape)
        y3 = self.DeBlock3(y3)
        #print("y3 shape:", y3.shape)

        #y2 = self.DeUp2(y3, x1_1)  # ()
        y2 = self.convTrans2(y3)
        #print("y2 shape:", y2.shape)
        y2 = self.DeBlock2(y2)
        #print("y2 shape:", y2.shape)

        y = self.endconv(y2)      # (1)
        #y = self.Softmax(y)
        if self.reversal_flag:
            y = grad_reverse(y)

        return y, torch.sigmoid(z_out)

class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        self.bn1 = nn.BatchNorm2d(128 // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(128 // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128 // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(128 // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = 0.5*x1 + 0.5*x

        return x1


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=1)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(out_channels*2, out_channels, stride=1, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        y = self.conv2(y)
        # y = y + prev
        #print("y shape:", y.shape)      #[8, 16, 32, 32]
        #print("x1 shape:", x1.shape)    #[8, 16, 16, 16]
        #print("prev shape:", prev.shape) #[8, 16, 64, 64]
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = 0.5*x1 + 0.5*x

        return x1




def TransBTS(dataset='breast2', _conv_repr=True, _pe_type="learned", _gr_flag=False):

    if dataset.lower() == 'breast2':
        img_dim = 256
        #num_classes = 4
    #print(dataset)
    #print(gr_flag)
    num_channels = 1
    patch_dim = 16
    aux_layers = [1, 2, 3, 4] ##??
    model = BTS(
        img_dim,
        patch_dim,
        num_channels,
        #num_classes,
        embedding_dim=128,
        num_heads=2,
        num_layers=4,
        hidden_dim=256,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
        # gr_flag=_gr_flag
    )

    return aux_layers, model


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        devide_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.rand((1, 1, 256, 256), device=devide_id)
        _, model = TransBTS(dataset='breast', _conv_repr=True, _pe_type="learned")
        model.to(devide_id)
        y = model(x)
        print("y shape:", y.shape)
        print("y min:", torch.min(y))
        print("y max:", torch.max(y))

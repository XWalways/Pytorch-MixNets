#from mixnet import MixNet
import models
from models import MixNet
from flops_counter import get_model_complexity_info
net = MixNet('s')
flops, params = get_model_complexity_info(net, (224,224), as_strings=False, print_per_layer_stat=False)
print("FLOPS: %.3fG"%(flops/1e9))
print("PARAMS: %.2fM"%(params/1e6))

from __future__ import absolute_import
from collections import OrderedDict


from ..utils import to_torch

def extract_cnn_feature(args, epoch, model, inputs, modules=None):
    model.eval()
    # with torch.no_grad():
    inputs = to_torch(inputs).cuda()
    if modules is None:
        if args.arch == 'cresnet50':
            if epoch == 0:
                outputs = model(inputs)
            else:
                outputs = model(inputs, apply_conststyle=True, is_test=True, apply_rate=0.5)
        else:
            outputs = model(inputs)
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())

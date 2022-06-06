import sys

import jsbsim
import torch
import torch.nn as nn

sys.path.append(str(jsbsim.get_default_root_dir()) + '/FCM/')


def get_rl(status, 
           property, 
           path:  str=None
           ) -> torch.tensor:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if path is not None:
        # model = torch.load(path, map_location=torch.device('cpu'))
        model = path
    else:
        # raise Exception("1")
        try:
            model = torch.load('./bestModel/Epoch.pt', map_location=torch.device('cpu'))
        except:
            from myModule.representation_learning.train import model
    # model.eval()

    status = status.to(device)
    property = property.to(device)

    # device_ids = [0]
    # model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    rep = model(status, property, 1)
    rep = rep.to(device)

    return rep

if __name__ == "__main__":
    x = get_rl(status=torch.rand([1, 9, 10, 50, 50]),
               property=torch.rand(3), 
               path='./FCM/bestModel/Epoch0acc1.0.pt'
               )

    print(x.size())

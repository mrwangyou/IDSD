import torch
import time

class IDSDRep():
    
    def __init__(self, env, model, device) -> None:
        self.env = env
        self.grid = torch.zeros([1, 9, 50, 50])
        self.model = model
        self.device = device

    def setStatus(
        self,
        ftpg,
        id:  "1|2",
    ):
        ego_x = self.env.getFdm(id).getProperty('positionEci')[1]
        ego_y = self.env.getFdm(id).getProperty('positionEci')[2]
        ego_s = torch.Tensor(self.env.getFdm(id).getProperty('velocity') + \
            self.env.getFdm(id).getProperty('pose')).squeeze(dim=0)  # A Tensor of size [1, 9]
        oppo_x = self.env.getFdm(id^3).getProperty('positionEci')[1]
        oppo_y = self.env.getFdm(id^3).getProperty('positionEci')[2]
        oppo_s = torch.Tensor(self.env.getFdm(id^3).getProperty('velocity') + \
            self.env.getFdm(id^3).getProperty('pose')).squeeze(dim=0)  # A Tensor of size [1, 9]

        if abs(oppo_x - ego_x) > ftpg * 24 or abs(oppo_y - ego_y) > ftpg * 24:
            # raise Exception("Plane out of range!", abs(oppo_x - ego_x), abs(oppo_y - ego_y), ftpg * 24)
            print("Plane out of range!")
            time.sleep(1)
            return -1

        self.grid[0, :, 25, 25] = ego_s
        self.grid[0, :, 25 + int((oppo_x - ego_x) // ftpg), 25 + int((oppo_y - ego_y) // ftpg)] = oppo_s

        # return self.grid

    def getStatus(
        self,
    ) -> torch.Tensor:
        return self.grid

    def getRl(
        self,
        id:  "1|2",
        ftpg=500,
    ):
        self.setStatus(ftpg, id)
        if id == 1:
            return self.model(self.getStatus().to(self.device), torch.Tensor([self.env.getDistance()] + self.env.getHP()).unsqueeze(0).to(self.device))
        elif id == 2:
            return self.model(self.getStatus().to(self.device), torch.Tensor([self.env.getDistance()] + self.env.getHP()[::-1]).unsqueeze(0).to(self.device))
        else:
            raise Exception("Plane {} doesn\'t exist!".format(id))


class Representation():

    def __init__(self, env) -> None:
        self.env = env
    
    def getRepresentation(self, modelType, model, device, id):
        if modelType == 'IDSD':
            self.rep = IDSDRep(self.env, model, device)
            return self.rep.getRl(id=id)

    def getStatus(self) -> torch.Tensor:
        return self.rep.getStatus()

    def getProperty(self, id) -> torch.Tensor:
        if id == 1:
            return torch.Tensor([self.env.getDistance()] + self.env.getHP())
        elif id == 2:
            return torch.Tensor([self.env.getDistance()] + self.env.getHP()[::-1])
        else:
            raise Exception("Plane {} doesn\'t exist!".format(id))


if __name__ == '__main__':
    pass
















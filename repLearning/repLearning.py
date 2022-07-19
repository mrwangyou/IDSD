import torch


class IDSDRep():
    def __init__(self, env, model, device) -> None:
        self.env = env
        self.grid = torch.zeros([1, 9, 100, 50, 50])
        self.model = model
        self.device = device

    def setStatus(
        self,
        nof,
        ftpg,
        id:  "1|2",
    ):
        ego_x = self.env.getProperty('positionEci', id)[0][1]
        ego_y = self.env.getProperty('positionEci', id)[0][2]
        ego_s = torch.Tensor(self.env.getProperty('velocity', id)[0] + \
            self.env.getProperty('pose', id)[0]).squeeze(dim=0)
        oppo_x = self.env.getProperty('positionEci', id ^ 3)[0][1]
        oppo_y = self.env.getProperty('positionEci', id ^ 3)[0][2]
        oppo_s = torch.Tensor(self.env.getProperty('velocity', id ^ 3)[0] + \
            self.env.getProperty('pose', id ^ 3)[0]).squeeze(dim=0) 
        print(self.env.getDistance())
        if abs(oppo_x - ego_x) > ftpg * 24 or abs(oppo_y - ego_y) > ftpg * 24:
            # raise Exception("Plane out of range!", abs(oppo_x - ego_x), abs(oppo_y - ego_y), ftpg * 24)
            # print("out of range")
            return -1

        self.grid[0, :, nof%100, 25, 25] = ego_s
        self.grid[0, :, nof%100, 25 + int((oppo_x - ego_x) // ftpg), 25 + int((oppo_y - ego_y) // ftpg)] = oppo_s

    def getStatus(
        self,
        nof,
    ) -> torch.Tensor:
        return self.grid[0, :, [i for i in range(nof % 10, 100, 10)], :, :].unsqueeze(0)

    def getRl(
        self,
        nof,
        id,
        ftpg=1000,
    ):
        self.setStatus(nof, ftpg, id)
        print("***{}".format(self.model(self.getStatus(nof).to(self.device), torch.Tensor([self.env.getDistance()] + self.env.getHP(0)).unsqueeze(0).to(self.device))))
        if id == 1:
            return self.model(self.getStatus(nof).to(self.device), torch.Tensor([self.env.getDistance()] + self.env.getHP(0)).unsqueeze(0).to(self.device))
        elif id == 2:
            return self.model(self.getStatus(nof).to(self.device), torch.Tensor([self.env.getDistance()] + self.env.getHP(0)[::-1]).unsqueeze(0).to(self.device))


class Representation():

    def __init__(self, env) -> None:
        self.env = env
    
    def getRepresentation(self, nof, modelType, model, device, id):
        if modelType == 'IDSD':
            self.rep = IDSDRep(self.env, model, device)
            return self.rep.getRl(nof, id=id)

    def getStatus(self, nof):
        return self.rep.getStatus(nof)

    def getProperty(self):
        return [self.env.getDistance()] + self.env.getHP(0)


if __name__ == '__main__':
    pass
















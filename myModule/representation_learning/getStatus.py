import torch

def setStatus(
    cnt:  int,
    ftpg:  float,
    oppo_x:  float,
    oppo_y:  float,
    ego_x:  float,
    ego_y:  float,
    oppo_s:  torch.Tensor,
    ego_s:  torch.Tensor
):
    global grid
    try:
        grid = grid
    except NameError:
        grid = torch.zeros([1, 9, 100, 50, 50])

    if abs(oppo_x - ego_x) > ftpg * 24 or abs(oppo_y - ego_y) > ftpg * 24:
        # raise Exception("Plane out of range!", abs(oppo_x - ego_x), abs(oppo_y - ego_y), ftpg * 24)
        print("out of range")
        return -1

    grid[0, :, cnt%100, 25, 25] = ego_s
    grid[0, :, cnt%100, 25 + int((oppo_x - ego_x) // ftpg), 25 + int((oppo_y - ego_y) // ftpg)] = oppo_s

def getStatus(
    cnt:  int
) -> torch.Tensor:
    global grid

    try:
        grid = grid
    except:
        raise Exception("Grid unexist!")

    return grid[0, :, [i for i in range(cnt % 10, 100, 10)], :, :].unsqueeze(0)

if __name__ == '__main__':
    pass

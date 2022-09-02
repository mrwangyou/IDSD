# IDSD, an <b>I</b>ntelligent <b>D</b>ecision <b>S</b>ystem for <b>D</b>ogfighting based on reinforcement learning

正在重构中的IDSD智能空战仿真平台与机动决策算法

### 准备工作

1. 如果需要使用Dogfight框架，需要在`./src/environments/dogfight/`下运行
    ```
    git clone https://github.com/harfang3d/dogfight-sandbox-hg2.git
    ```
    并将生成的文件夹重命名为`dogfight_sandbox_hg2`。

2. 如果需要使用FlightGear可视化，请在FlightGear启动时采用如下参数
    ```
    --fdm=null --native-fdm=socket,in,60,,5550,udp
    ```

    如需同时对两架飞机进行可视化，请在`..(JSBSim)/data_output/`下复制两份`flightgear.xml`文件，并将两个文件分别命名为`flightgear{1/2}.xml`，将`flightgear2.xml`第18行的`5550`修改为`5551`，并在FlightGear启动时分别使用
    ```
    --multiplay=out,10.127.0.0.1,5000 --multiplay=in,10.127.0.0.1,5001 --callsign=Test1
    ```
    与
    ```
    --multiplay=out,10.127.0.0.1,5001 --multiplay=in,10.127.0.0.1,5000 --callsign=Test2
    ```
    参数。

3. 如果需要打印每一帧的飞行器状态，可以修改飞行器文件，详情待续。

4. 未完待续

### 备注

1. 本框架关注于（强化学习）算法，故不考虑雷达、视距、数据噪声等特征；

2. 根据JSBSim与dogfight-sandbox-gh2的许可，本项目遵循LGPL2.1 License（存疑，但是应该是要开源的）。

### 未来工作

1. 太多了以至于下笔不知道从哪里写起

## 阵线全部迁移至最新Git repo（虽然还没开始建）

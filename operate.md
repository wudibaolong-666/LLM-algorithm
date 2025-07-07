[toc]

### SSH

~~~python
# Read more about SSH config files: https://linux.die.net/man/5/ssh_config
Host server6-gfkd
    HostName 10.134.2.23
    User root
    Port 31200
    # secret 1953
    
Host ray-server-5
    HostName 10.134.2.23
    # HostName 10.135.38.241
    User root
    Port 31241
    # secret 8848

Host ray-server-6
    HostName 10.134.2.23
    User root
    Port 31240
    # secret 8848
~~~

### Logic-RL-main

1. 进入server6-gfkd

2. 更改conda环境

   `source /home/data/.bashrc`

3. 查看虚拟环境

   `conda env list`

4. 进行虚拟环境rlhf

   `conda activate rlhf`

5. 进入目标文件夹

   `cd data/Logic-RL-main/examples/ppo_trainer`

6. 查看wandb秘钥地址

   `pip show wandb`

7. 添加wandb秘钥

   `python "/home/data/miniconda3/envs/rlhf/lib/python3.9/site-packages/wandb" login`

8. 运行文件`run_qwen2.5_1.5b.sh`

   `sh run_qwen2.5_1.5b.sh`

### unsloth

1. 进入server6-gfkd

2. 更改conda环境

   `source /home/data/.bashrc`

3. 查看虚拟环境

   `conda env list`

4. 进行虚拟环境rlhf

   `conda activate unsloth`

5. 进入目标文件夹下

   `cd data/unsloth-main`

6. 运行`run_unsloth.sh`文件

   `sh run_unsloth.sh`

7. 运行`nvitop`查看显卡进程

### OpenRLHF-main

1. 进入 ray-server-6 

2. 进入文件`OpenRLHF-main/examples`

3. 运行`ray_start.sh`

   得到输出`ray start --address='10.42.17.120:6379'`

4. 进入 ray-server-5，复制刚刚得到的输出到终端

5. 进入 ray-server-6 ,  运行`debug_ray_ppo_lzy.sh`

6. 查看多进程帮助信息 `ray :: help`

7. 关闭多进程 `ray stop`




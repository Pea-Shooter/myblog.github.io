---
layout: post
title: How to configure PyCharm with remote docker environment? 
date: 2018-08-07 14:00:00.000000000 +09:00
tags: docker
---

[TOC]

> This is  a tutorial about how to configure PyCharm with remote docker environment (especially for Python). The precondition of  this tutorial is that you should have basic knowledge of **Docker** and **Linux**, and your PyCharm should be **professional** version, not the community.

# 1. Create your docker container

```bash
sudo nvidia-docker run -it -p [host_port]:[container_port](do not use 8888) --name:[container_name] [image_name] -v [container_path]:[host_path] /bin/bash
```

For example:

```bash
sudo nvidia-docker run -p 5592:5592 -p 5593:5593 -p 2022:22 --name="liuzhen_tf" -v ~/workspace/liuzhen/remote_workspace:/workspace/liuzhen/remote_workspace -it tensorflow/tensorflow:latest-gpu /bin/bash
```

Now we are in our docker container environment.

#  2. SSH server configuration 

First, install `openssh-server`

```bash
$ apt update
$ apt install -y openssh-server
```

then, create a folder to configure openssh server

```bash
$ mkdir /var/run/sshd
```

here is the configuration of openssh server:

```bash
$ echo 'root:your_passwd' | chpasswd
# Root password was changed with your_passwd
$ sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
$ sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
$ echo "export VISIBLE=now" >> /etc/profile
```

restart ssh service to activate the configuration above.

```bash
$ service ssh restart
```

now, test which port of the host was forwarded to 22 port.

```bash
$ sudo docker port [your_container_name] 22
# if the configuration work, you will see the output below
# 0.0.0.0:2022
```

Finally, test whether you can connect to the docker container with ssh.

```bash
$ ssh root@[your_host_ip] -p 2022
# the password is what you set above
```



# 3. PyCharm deployment configuration

Follow the `Tools > Deployment > Configuration` , add a new `SFTP` server. Here you should pay attention that the port is what you set to mapping to you docker container's 22 port number(Here is 2022).

In detail, you can follow [this blog](https://www.cnblogs.com/xiongmao-cpp/p/7856596.html) .

# 4. Remote interpreter configuration

Open the `File > Setting > Project > Project Interpreter` page, add a new remote interpreter.

In detail, you can follow [this blog](https://www.cnblogs.com/xiongmao-cpp/p/7856596.html) .

After you add the remote interpreter, you should wait for about 30 minutes to complete the configuration.



**If you have any problem , contact me with my email.**
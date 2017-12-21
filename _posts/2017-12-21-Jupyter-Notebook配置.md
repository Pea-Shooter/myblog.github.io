---
layout: post
title: Jupyter notebook configuration
date: 2017-12-21 11:00:00.000000000 +09:00
tags: python
---



## Remote server configuration

1. Install jupyter in your server

   ```shell
   pip3 install jupyter 
   ```

2. Generate configuration file

   ```powershell
   jupyter notebook --generate-config
   ```

3. Generate password

   ```python
   # open your jupyter notebook with `jupyter notebook` and create a python(3) notebook
   from notebook.auth import passwd
   passwd()
   Enter password:
   Verify password:
   out[2]: 'sha1:sdhflalsfqoupwru10747515qlhglasjhfauy91uaf'
   # then copy the output string
   ```

4. Modify the default profile

   ```shell
   c.NotebookApp.ip='*' # allow all ip access your jupyter
   c.NotebookApp.password = u'sha1:sdhf...the copied string'
   c.NotebookApp.open_browser = False # Disable automatic open browser
   c.NotebookApp.port =8888 # assign a port
   c.NotebookApp.notebook_dir = u'/Users/liuzhen/jupyter' # Set default directory
   ```

5. Start your jupyter notebook

   ```shell
   jupyter notebook
   jupyter notebook --allow-root # if necessary
   ```

## Jupyter Theme

you can also set a theme for your jupyter

1. Install `jupyterthemes`

   ```shell
   pip3 install jupyterthemes
   ```

2. Usage

   ```shell
   jt	[-h] [-l] [-t THEME] [-f MONOFONT] [-fs MONOSIZE] [-nf NBFONT]
     	[-nfs NBFONTSIZE] [-tf TCFONT] [-tfs TCFONTSIZE] [-dfs DFFONTSIZE]
   	[-ofs OUTFONTSIZE] [-mathfs MATHFONTSIZE] [-m MARGINS]
   	[-cursw CURSORWIDTH] [-cursc CURSORCOLOR] [-cellw CELLWIDTH]
   	[-lineh LINEHEIGHT] [-altp] [-altmd] [-altout] [-P] [-T] [-N] [-vim]
   	[-r] [-dfonts]
   ```

   some optional arguments:

   ```shell
   jt -h # get help
   ```

   ```shell
   jt -l # list available themes
   Available Themes:
      chesterish
      grade3
      gruvboxd
      gruvboxl
      monokai
      oceans16
      onedork
      solarizedd
      solarizedl
   ```

   ```shell
   jt -t onedork # set a theme
   ```

   ```shell
   jt -r # reset default theme
   ```

   reference:[jupyter-themes](https://github.com/dunovank/jupyter-themes)

3. run your jupyter notebook

   After select a theme,use the command `jupyter notebook` to start your jupyter,and you can change your theme during the runtime, just refresh it after your modification.
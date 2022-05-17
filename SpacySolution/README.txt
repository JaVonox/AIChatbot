python 3.10 and pip are required to use this software.

The main.py file depends on a number of different libraries to run correctly.
These libraries can be found in the requirements.txt file.

To quickly install the library necessary, use the following command:

pip install -r requirements.txt

(this may take some time as there are a lot of required files. Also ensure that this is referencing
the appropriate directory. Depending on the IDE used, a virtual enviroment may need to be 
created to use these libraries - if the software does not run, ensure a virual enviroment has been
initialised and is used when running the software. For pycharm, this is described at 
https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#env-requirements)

Most likely, this means that some form of python enviroment will be required to run this software -
I personally used pycharm for this, and so I atleast know this approach works. 

If while installing packages, you recieve a "memory error", there is a decent chance its because
of the large files. If this occurs, append the argument "--no-cache-dir" to the install command as such

pip install -r requirements.txt --no-cache-dir


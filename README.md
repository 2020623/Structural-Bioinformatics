# How to run the script

Suppose you have on the Desktop:
   - a python script, called $\color{red}{\text{algo.py}}$; <font color = 'red'> *algo.py* text</font>
   - an example directory called $\color{blue}{\text{protein}}$ which contains:
       - the  **pdb** directory;
       - the  **edges** directory.


Now let's create a new directory, called $\color{green}{\text{new}}$, and insert in it $\color{red}{\text{algo.py}}$ and the directory $\color{blue}{\text{protein}}$.

In order to run the script is necessary:
1. Open Anaconda Prompt (on Windows) or Command Prompt (on Linux);
2. Set in the directory $\color{green}{\text{new}}$ with:

<code> (base) C:\Users\name> cd Desktop/new</code>

3. Run the script by writing on the command line:

<code> (base) C:\Users\name\Desktop\new> python algo.py protein/pdb/snapshot_0.pdb protein/edges protein </code>
    
Where:
- <code>./snapshot_0.pdb</code> is just an example name of the first pdb file to pass to the procedure to extract the residues list;
- <code>protein</code> (the last argument to pass) indicates the folder where the output directory will be created.


4. After running the script it will automatically create an $\color{orange}{\text{output}}$ folder containing all the files returned by the procedure. 


**It is necessary to remember that inside $\color{orange}{\text{proteins}}$ there must not be other folders called $\color{blue}{\text{output}}$.**

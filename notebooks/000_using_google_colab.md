# Using Google Colab for Notebook Development

**Colab** is a Jupyter Notebook service that allows you to work on `.ipynb` files through Google Drive. 

>A **Jupyter Notebook** (file extension `.ipynb`) is a document which allows for seamless integration of Python code with Markdown text formatting, to allow for an intuitive programming pipeline. It is organized in cells, with each cell containing either code or Markdown. The outputs of each code cell lay below the respective cell. Also, the code is continuous from cell to cell; variables are accessible in all cells lower than their definition cell. Jupyter Notebook files are ideal for data analysis due to their presentability. Check out `/notebooks/Intro.ipynb` at https://jupyter.org/try-jupyter/lab/.

# Installation

To get started with colab, head to Google Drive, and select `New`, `More`, `Connect More Apps`.

![enter image description here](https://github.com/pmeshkov/Colab_Tutorial/blob/main/Google_colab_tutorial/Screenshot%202024-06-27%20at%2010.46.41%E2%80%AFAM.png?raw=true)

Search for *Colaboratory*, and select the first option.

![enter image description here](https://github.com/pmeshkov/Colab_Tutorial/blob/main/Google_colab_tutorial/Screenshot%202024-06-27%20at%2010.48.35%E2%80%AFAM.png?raw=true)

Install *Colaboratory*.

![enter image description here](https://github.com/pmeshkov/Colab_Tutorial/blob/main/Google_colab_tutorial/Screenshot%202024-06-27%20at%2010.49.42%E2%80%AFAM.png?raw=true)

Now that **Colab** is installed, it is seamlessly integrated into your Google Drive. All you need to do now is select `New`, `More`, `Google Colaboratory` to get started on a new `.ipynb`.

![enter image description here](https://github.com/pmeshkov/Colab_Tutorial/blob/main/Google_colab_tutorial/Screenshot%202024-06-27%20at%2011.19.01%E2%80%AFAM.png?raw=true)

Select `+ Code` to add a runnable Python cell. Select `+ Text` to add a Markdown cell. Use shortcut `Shift` + `Return` or `Shift` + `Enter
` to run either cell; a code cell will output the Python interpreted result below the cell, and a Markdown cell will turn into its document form. 

![enter image description here](https://github.com/pmeshkov/Colab_Tutorial/blob/main/Google_colab_tutorial/Screenshot%202024-06-27%20at%2011.27.18%E2%80%AFAM.png?raw=true)

# Unique Features

You can call Bash commands directly in the code cells, by typing `!` before a command. Install Python libraries with `!pip install [library]`.

![enter image description here](https://github.com/pmeshkov/Colab_Tutorial/blob/main/Google_colab_tutorial/Screenshot%202024-06-27%20at%2011.41.53%E2%80%AFAM.png?raw=true)

Google Colab also offers generative AI helper features. To use them, you need to check the consent box, which you can find by selecting `Tools` and then `Settings`. 

 ![enter image description here](https://github.com/pmeshkov/Colab_Tutorial/blob/main/Google_colab_tutorial/Screenshot%202024-06-27%20at%2011.49.50%E2%80%AFAM.png?raw=true)

You can use **Gemini** to help answer questions, as well as debug your code (select `Explain Error` when facing an error).

![enter image description here](https://github.com/pmeshkov/Colab_Tutorial/blob/main/Google_colab_tutorial/Screenshot%202024-06-27%20at%2011.51.28%E2%80%AFAM.png?raw=true)

Colab even has an inline generative AI code completion feature, which sometimes completes your code just based on variable names!

![enter image description here](https://github.com/pmeshkov/Colab_Tutorial/blob/main/Google_colab_tutorial/Screenshot%202024-06-27%20at%2012.30.11%E2%80%AFPM.png?raw=true)

Finally, let's talk about importing files from Google Drive. Before you begin to import files, you first must mount your notebook to your drive. You can do that with:

`from google.colab import drive`

`drive.mount(/content/drive)`

And then you need to allow the notebook to access your drive.

![enter image description here](https://github.com/pmeshkov/Colab_Tutorial/blob/main/Google_colab_tutorial/Screenshot%202024-06-27%20at%2012.56.57%E2%80%AFPM.png?raw=true)

Once connected, you can navigate your Google Drive directories using:

`import os`

`os.chdir(' [path to you dir] ')`

The `os.chdir()` command operates the same way that the `cd` command (unix, windows, or macOS) would navigate directories in terminal. Now you can load and utilize files just as you would using Python elsewhere.

![enter image description here](https://github.com/pmeshkov/Colab_Tutorial/blob/main/Google_colab_tutorial/Screenshot%202024-06-27%20at%2012.58.03%E2%80%AFPM.png?raw=true)

>For more information on **Colab** check the official tutorial from Google https://g.co/kgs/aXJwnWt

> Written with [StackEdit](https://stackedit.io/).

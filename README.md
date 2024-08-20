# AKKODIS get started with chatbots
@author: Jakub Rosol, (jakub.rosol@akkodis.com)

This guide will take you from zero to hero in topic of AI language models integrations into
python programming. Follow this tutorial step by step and become AKKODIS AI developer.

## Pre-requisities

 - **Python** - The more you know, the better. But even our junior colleagues were able to go through this course 
with some guidance. Check that you have python version >= 3.10


 - **API KEY** - you need one of the following:
   - **OPEN AI api key** - You can create an openai account and buy some token credit for their API
   . On their website you can generate key for your profile and use it in this project.
   - **AKKODIS api key** - You need access to a LLM api. Since we are AKKODIS, you can use our 
   own GPT deployment. Go to https://cld-developer.akkodis.com/ and sign in with your company credentials.
   Then you will need access to the APIs and for that you can write to Mr. Consigny
   (nicolas.consigny@akkodis.com). Save the key somewhere safe like password app. **!!!DO NOT SHARE THE KEY WITH ANYONE!!!**

## Initial Setup

### Copy source
Download the repository if you haven't already. (It is up to you where you want to save the project,
but you will need sufficient permissions to install python environment.)

Either by git clone:
```bash
git clone <repo link>
```

Or just download the project as folder from somewhere like Sharepoint
```bash
https://github.com/RetardCZE/Hackaton2024
```

### Create environment

To separate this guide from other projects, we will work in a virtual python environment. This step
varies for different platforms, so you might need to find your own solution.

**Linux example (should work for Windows cmd too)**:
```bash
# Go to the project directory
cd </path/to/the/project/>

# Create the environment (usual names are .venv, .env, __venv__)
python -m venv <environment_name>
```
Do not forget to activate the environment when you work on the project. We recommend using IDE like pycharm,
where you can link the environment to the project, and it will be used automatically.

(If you return to already initialized project and you ModuleNotFound indicating that some python
packages are missing, you may have forgotten to activate the environment.)

### Install requirements
To run provided examples, you need various python packages.
If you want to understand what you need better. I would recommend installing them one by one
as you need them for the tasks.

If you don't want to bother with the installation, you can install all needed packages with:
```bash
# in the project directory
pip install -r requirements.txt
```
This will install all packages in versions used during guide development.

### Api key preparation
Implementation uses [akkodis_clients.py](AI_Tutorial/akkodis_clients.py), which loads api keys and 
api provider from [conf.py](AI_Tutorial/conf.py). As you might be tempted to write 
your api key directly in the file. It is ignored by git and only [conf.py.template](AI_Tutorial/conf.py.template)
is provided, so you have to copy or rename it to remove the .template suffix.

You can define 3 environmental variables:
`AKKODIS_API_KEY`, `OPENAI_API_KEY`, `PROVIDER`, which are 
then loaded by [conf.py](AI_Tutorial/conf.py). If you don't know how to define 
environmental variable, you can write the information directly to the file (but
it is not a good practice).

My recommendation is to place it as string in the code for the tutorial and return to environmental variables
once you implement your own project. (For example in pycharm you can set variable for a run 
configuration, which can be useful when you launch your project with something like run.py)

## The tutorial
The tutorial will take you through 9 tasks where you will learn how to use the API to
generate text and embeddings. 


First take a quick look at [akkodis_clients.py](AI_Tutorial/akkodis_clients.py) where you can see how to
initialize client for communication with api using openai package. 
You won't reimplement the code in this 
tutorial, **but you will implement it differently in your productive tasks.**

Then proceed to task 1 where you will test if your API connections works.
After that you can go through tasks 2-8 where you should always start with relevant
[main.py] and try to fill in missing code until it does what is should. Each task is 
provided with [solution.py] file to compare with your solution or to be used as hint if 
you get stuck.

Try to do the tasks on your own. You can always copy the code to chatGPT and it will most
likely finish the task for you, but you will learn nothing.

Final Task 9 is only in form of solution as it can have quite variable implementations. Read it
and use it as starting point for your own project. 

As a final test you should make your own project. For example on our AI camp
we did tool for summarization of reported work, assistant for ticketing systems
(answer repetitive questions, redirect badly targeted tickets), AI enhanced interface
for physics modeling engine. Of coarse they were demo versions, but we did them in one day.
Just be creative.

**LAST REMINDER**
 - DO NOT SHARE THE API KEY
 - DO NOT UPLOAD THE API KEY TO GIT
 - DO NOT UPLOAD THE CODE WITH API KEY TO ANY PLATFORM (chatGPT)

# Happy Coding!
### from Akkodis CZ team




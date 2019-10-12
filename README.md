# Weak CN star Analysis in Andromeda M31

## Table of Contents

  * [Requirements and Setup](#requirements-and-setup)
  * [Files](#files-in-repository)  
  * [Troubleshooting](#troubleshooting) 

## Requirements and Setup 
    
   ### Anaconda
    Download Anaconda from the website below. Make sure to install Python 3.7 version! 
[Anaconda link](https://www.anaconda.com/distribution/) 
    
  ### Using Jupyter
    Open up Anaconda Navigator and type 'jupyter notebook'. This launches jupyter on your web browser.
    Alternatively use terminal to cd to directory and open jupyter from there by typing 'jupyter notebook'

**If you are new to coding with python in Jupyter, I suggest taking a look at this Python tutorial links:**

1. Please make sure you have completed the Anaconda installation of Python 3 on your computer. 
    Link to instructions for Anaconda installation and other pre-tutorial preparatory steps:

[Installation and pre-tutorial](https://drive.google.com/open?id=13O_K943llhgoTA4Puo32s4xAPtA2NL3kCpl4pHoTobI)

2. Website that contains the blank Jupyter notebooks that we will use for the PyaR tutorial
   (download and unpack item 5 named JupyterNotebooksBlank,zip) and other relevant materials:

[Python notebooks](http://www.ucolick.org/~raja/rm4/Astro/)

3. Main PyaR Google doc link:

[Pyar link](https://docs.google.com/document/d/16QuhwolhX0URjpyPZ7sfWy107aBuvq3E9-LPyT8Nz68/edit?usp=sharing)
   
   #### Install Jupyter Extensions
    Instructions can be found below:
   [Extensions](https://ndres.me/post/best-jupyter-notebook-extensions/)

    Recommended Extensions: collapsible headings, table of contents, move selected cells

  ### Using Github
  
  #### 1. Set up a Git repository on your laptop
    
    1. Go to terminal and cd to the spot you would like the WeakCN2019 repository (folder)
    2. Then type: git clone https://github.com/rachelraikar/WeakCN2019.git into your terminal. 
        This will copy the repository as a folder onto your laptop with the files
        
  #### 2. Set up Testbranch
   Be sure to make changes to the testbranch. Keep the master branch as a backup. Update master when needed.
   Here are basic instructions to get started :
    
    Set up a stream to GitHub (So that you can push/pull changes):
      git push --set-upstream origin Testbranch
   
    Check which branch you are in: git branch
    Switch branches to desired branch: git checkout <branch>
    Create a new branch: git branch <name_of_your_new_branch>
    
  #### How to pull/push changes
  
   Always check which branch you are in before pushing/pulling. 
   You should be in 'Testbranch' unless pushing to Master. If not, set a stream to the Testbranch and move into it.
   
    Pull Changes:  

        To pull any changes made by others, use terminal to navigate to directory and type git pull.
   
    Push Changes:
       
        After making any changes on the notebook, save and checkpoint. These changes will be saved locally. 
        Then, to push this to Github, navigate to the directory on terminal and type: 
                 git add 'the file name
                 git commit -m 'your message/comments'  (-a -m instead of -m if not mac user)
                 git push
    
        The change should now be visible on github."

## Files In Repository

  **Automation Version 1.4.1**

    1. SIP 2016 code by interns Anika Kamath, Atmika Sarukkai, Alyssa Sales. Mentor: Puragra GuhaThakurta.
    2. Data cleaning and plotting Weak CN stars on CMDs. 

  **Automation Version 2.5.3**

    1. SIP 2018 code by interns Alexandra Masegian and Arya Maheshwari. Mentors: Rachel Raikar,Anika Kamath
    2. Created Carbon template spectrum and graphed scores against it. 
    3. Kernel density estimations for classification.

  **Automation Version 2.6.1**
      
    1. SIP 2018 code by Alexandra Masegian and Arya Maheshwari. Mentors: by Rachel Raikar,Anika Kamath
    2. Created Carbon template spectrum and graphed scores against it. 
    3. Kernel density estimations for classification, 
    4. Slope calculations

  **Automation Version 2.7 test**
      
    1. SIP 2019 code by Antara Bhattacharya, Suhas Kotha, Allison Chang. Mentored by Rachel Raikar
    2. Created Weak CN Template and performed analysis in comparison to Carbon Template.
    3. Machine-classification of Weak CN and Carbon stars based  on their positions on 3 graphs:
        * Score against Weak CN Template vs Score against Carbon Template (both unmodified)
        * Score against scores against modified-diluted carbon template vs dilution factor
        * Slope 4-3 vs Slope 2-1 of the 'W' range.

  **Automation Version 3.0**
    
    1. Combined and cleaned up code from 2.5.3 (Arya's code) and 2.6.1 (Alexandra's code).

  **Automation Version 4.0**
      
    1. Code by Rachel Raikar and Antara Bhattacharya. 
    2. Classification of weak CN stars based on their position on CMDs with     
      distance from the head of the plot of scores against undiluted templates used as a classification metric.
   
   **CSV-Dictionaries**
  
    Folder containing folders of dictionaries from all methods of analysis:
      1. Modified Diluted 
        * Contains both Carbon and Weak CN template scores, dilution factor (c), and slope adjustment (s)
        * For full W and 1U range
      2. Modified Undiluted
        * Contains both Carbon and Weak CN template scores, dilution factor (c), and slope adjustment (s)
        * For full W range
      3. Unmodified
        * Contains both Carbon and Weak CN template scores
        * For full W range
      4. Slopes
        * Contains 4 slopes of W range for all stars
        
   **validindices**
   
   **starVals2**
   
     
  
## Troubleshooting
  
    Google doc with resources for help with Python/Github problems 
[Click here for more help.](https://docs.google.com/document/d/1nbBvIYcEp0FrCOEeOlo-bxkvCmlRNhHeFeaXnpO_46g/edit?ts=5d0d0d6f)
  

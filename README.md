# Weak CN star Analysis in Andromeda M31
  
## Requirements/ Setup
  ### Installing Anaconda
    Download Anaconda from the website below. Make sure to install Python 3.7 version! 
[Anaconda link](https://www.anaconda.com/distribution/) 
    
  ### Using Jupyter
    Open up Anaconda Navigator and type 'jupyter notebook' into it. This launches jupyter on your web browser.
    Alternatively use terminal to cd to directory and open jupyter from there by typing 'jupyter notebook'

  ### How to set up a Git repository on your laptop
    
    1. Go to terminal and cd to the spot you would like the WeakCN2019 repository (folder)
    2. Then type: git clone https://github.com/rachelraikar/WeakCN2019.git into your terminal. 
        This will copy the repository as a folder onto your laptop with the files\n",
    
  ### How to pull/push changes
   #### Pull Changes 
    To pull any changes made by others, use terminal to navigate to directory and type git pull.
   
   #### Push Changes
       
    After making any changes on the notebook, save and checkpoint. These changes will be saved locally. 
    Then, to push this to Github, navigate to the directory on terminal and type: 
             git add 'the file name
             git commit -m 'your message/comments'  (-a -m instead of -m if not mac user)
             git push
    
    The change should now be visible on github."

## Files In Repository

  **Automation Version 1.4.1**

    1. SIP 2016 code by interns Anika Kamath, Atmika Sarukkai, Alyssa Sales. Mentored by Puragra GuhaThakurta.
    2. Data cleaning and plotting weak CN stars on CMDs. 

  **Automation Version 2.5.3**

    1. SIP 2018 code by interns Alexandra Masegian and Arya Maheshwari. Mentors: Rachel Raikar, Anika Kamath.
    2. Creating a carbon template spectrum and graphing scores against it, kernel density estimations for classification.

  **Automation Version 2.6.1**
      
    1. SIP 2018 code by Alexandra Masegian and Arya Maheshwari. Mentors: by Rachel Raikar, Anika Kamath.
    2. Creating a carbon template spectrum and graphing scores against it, kernel density estimations for classification, 
    slope calculation.

  **Automation Version 2.7 test**
      
    1. SIP 2019 code by Antara Bhattacharya, Suhas Kotha, Allison Chang. Mentored by Rachel Raikar,
    2. Machine-classification of weak CN and carbon stars based  on their positions on graphs of scores against undiluted and 
      diluted carbon and weak CN templates and slope magnitudes.

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
     
  
## Troubleshooting
  
    Google doc with resources for help with Python/Github problems 
[Click here for more help.](https://docs.google.com/document/d/1nbBvIYcEp0FrCOEeOlo-bxkvCmlRNhHeFeaXnpO_46g/edit?ts=5d0d0d6f)
  

# Deep-Learning
Useful Notes from the Deep Learning Specialization
--------------------------------------------------------------------------------------------------------------
# Guidelines that can assist you in your ML or DL Project
Note:-
- For this case we will be considering the problem involved in the pipeline of autonomous driving.
- The problem is identifying cars,signs and pedestrians from the images taken by the camera of your car. 

## First Step :-

-- Divide the dataset into Train,Train-Dev,Dev and Test set.
-- Note:
 If you have a very large dataset say 100,000 downloaded from internet and you have 10,000 example captured 
 by the camera[let's say J] of your car then the right choice would be to give 5,000 examples [J] to the training 
 datset set and divide the remaining among Dev and Test sets. Train-Dev set is from the trainig set distribution.
 

## Second Step

-- Train your initial model on this dataset.
-- Instead of training one version of a model, train multiple versions and then record their performance.
-- Check their Performance on the Train-Dev,Dev and Test sets.
-- Consider the model that is most likely to satisfy the threshold.
-- Consider the Bayesian(~Human Level) erorr.
-- Put all the sets in one column and their accuracy in another column.
-- Highlight those sets that are very diverging.


## Third Step

Consider Following Steps:-
-- Change the Evaluation Metric : If the model chosen perform worse than the model we neglected at the time of 
    threshold.
-- Pick a reasonable amount of mislabeled examples and then create a table of how many examples were misclassified
    from each category of misclassified images. i.e., Error Evaluation
-- Choose the category to improve so that it affects the accuracy most as compared to other categories.
-- If there is a huge difference between Train-Dev and Dev set error then consider applying techniques of Mismatched
    Training and Dev sets.


## Bonus Steps
-- Try using transfer learning for better results if you don't have enough dat for your features.
-- Consider combining Component and End to End based learning to get good results even if you don't have a huge 
    dataset.
-- Consider Reading the case Studies of Peacetopia and Autonomous Driving by Deeplearning.ai . You can find it 
    online as well as in Machine Learning--> Project_Ideas.
-- [Private: Consider reading your notes in Book3 for understanding how to tackle each of the above steps.]

##AI for everyone
-- Consider AI for replacing tasks not Jobs.
-- Don't compete with tech giants, do whatever your country is good at doing
-- Data science is about extracting meaningful information out of data.
-- Adversarial attacks on AI.
-- Before making samethong, make sure that AI can do that.
-- Workflow of ML projects.
-- Technical and Business Diligence
-- Start with a project to gain some momentum and then move onto bigger ones.
-- AI transmission playbook
-- For brainstorming AI projects, first consult with the person that is a specialized person in that field.(or business)
-- Laxk of explanability can force people not to believe like in case of X-ray scans.
